"""Scoring utilities for the models-lstm package."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import torch  # type: ignore[import-not-found]
from torch import Tensor  # type: ignore[attr-defined]

from . import data as data_module
from .data import ContractRecord, load_contract_dataframe
from .train import (
    AnomalyDetectorModel,
    _compute_file_sha256,
    _resolve_device,
    log_event,
    set_deterministic_mode,
)


SCORE_COLUMNS: Sequence[str] = (
    "timestamp_utc",
    "uid",
    "session_id",
    "op_category",
    "z",
    "z_hat",
    "s_cls",
    "s_time",
    "S",
    "flag_cls",
    "flag_dt",
)
DEFAULT_WEIGHT_CLS: float = 0.5
DEFAULT_WEIGHT_TIME: float = 0.5


@dataclass
class _Checkpoint:
    """Container for loaded checkpoint artefacts."""

    model_state: MutableMapping[str, Tensor]
    config: Mapping[str, Any]
    encoder_vocab: Sequence[str]


@dataclass
class ScoreResult:
    """Result payload produced after scoring a dataset."""

    rows: List[Dict[str, Any]]
    seed: int


def _load_checkpoint(model_path: Path) -> _Checkpoint:
    """Load the serialized checkpoint produced during training."""

    payload = torch.load(model_path, map_location="cpu")  # type: ignore[arg-type]
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint file must contain a mapping of artifacts.")

    try:
        model_state = payload["model_state"]
        config = payload["config"]
        encoder_vocab = payload["encoder_vocab"]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            "Checkpoint is missing one or more required keys: "
            "model_state, config, encoder_vocab"
        ) from exc

    if not isinstance(model_state, MutableMapping):
        raise ValueError("model_state in checkpoint must be a mutable mapping")
    if not isinstance(config, Mapping):
        raise ValueError("config in checkpoint must be a mapping")
    if not isinstance(encoder_vocab, Iterable):
        raise ValueError("encoder_vocab in checkpoint must be iterable")

    vocabulary = list(str(category) for category in encoder_vocab)
    return _Checkpoint(model_state=model_state, config=config, encoder_vocab=vocabulary)


def _load_thresholds(metrics_path: Path) -> Mapping[str, Any]:
    """Read threshold definitions from metrics.json if available."""

    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Failed to parse metrics file: {metrics_path}") from exc

    thresholds = data.get("thresholds") if isinstance(data, Mapping) else None
    if isinstance(thresholds, Mapping):
        return thresholds
    return {}


def _prepare_records(
    records: Sequence[Mapping[str, object]], *, clip_value: float
) -> List[ContractRecord]:
    """Normalize raw CSV rows into typed records with robust statistics."""

    prepared = data_module._prepare_records(records)
    sorted_records = sorted(
        prepared,
        key=lambda record: (record.uid, record.session_id, record.timestamp_utc),
    )
    delta_values = data_module._compute_delta_seconds(sorted_records)
    _, z_clipped, _ = data_module._compute_robust_z(delta_values, clip_value=clip_value)

    for record, clipped in zip(sorted_records, z_clipped):
        record.z_clipped = float(clipped)

    return prepared


def _build_model(checkpoint: _Checkpoint) -> AnomalyDetectorModel:
    """Instantiate the anomaly detector using checkpoint metadata."""

    config = checkpoint.config
    try:
        embed_dim = int(config["embed_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_layers = int(config["num_layers"])
        dropout = float(config.get("dropout", 0.0))
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            "Checkpoint configuration lacks required hyperparameters."
        ) from exc

    model = AnomalyDetectorModel(
        num_categories=len(checkpoint.encoder_vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint.model_state)
    return model


def _score_session(
    model: AnomalyDetectorModel,
    records: Sequence[ContractRecord],
    *,
    category_to_index: Mapping[str, int],
    thresholds: Mapping[str, Any],
    weight_cls: float,
    weight_time: float,
    device: torch.device,
) -> None:
    """Score a single session worth of records in-place."""

    if not records:
        return

    for record in records:
        record.z_hat = 0.0  # type: ignore[attr-defined]
        record.s_cls = 0.0  # type: ignore[attr-defined]
        record.s_time = 0.0  # type: ignore[attr-defined]
        record.S = 0.0  # type: ignore[attr-defined]
        record.flag_cls = 0  # type: ignore[attr-defined]
        record.flag_dt = 0  # type: ignore[attr-defined]

    if len(records) < 2:
        return

    indices: List[int] = []
    z_values: List[float] = []
    for record in records:
        category = record.op_category
        if category not in category_to_index:
            raise ValueError(
                f"Unknown op_category encountered during scoring: {category}"
            )
        indices.append(category_to_index[category])
        z_values.append(float(record.z_clipped))

    op_tensor = torch.tensor(
        [[index + 1 for index in indices[:-1]]], dtype=torch.long, device=device
    )
    z_tensor = torch.tensor([z_values[:-1]], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        op_probs, z_hat = model(op_tensor, z_tensor)

    op_probs_cpu = op_probs.squeeze(0).detach().cpu()
    z_hat_cpu = z_hat.squeeze(0).detach().cpu()

    per_category = (
        thresholds.get("per_category", {}) if isinstance(thresholds, Mapping) else {}
    )

    for position in range(1, len(records)):
        record = records[position]
        previous_index = position - 1
        actual_index = indices[position]
        probability_vector = op_probs_cpu[previous_index]
        probability = float(probability_vector[actual_index].item())
        probability = max(0.0, min(1.0, probability))

        z_prediction = float(z_hat_cpu[previous_index].item())
        residual = abs(z_prediction - z_values[position])

        s_cls = 1.0 - probability
        s_time = residual
        combined = weight_cls * s_cls + weight_time * s_time

        tau_hi: float | None = None
        if isinstance(per_category, Mapping):
            category_thresholds = per_category.get(record.op_category)
            if isinstance(category_thresholds, Mapping):
                tau_value = category_thresholds.get("tau_hi")
                if tau_value is not None:
                    tau_hi = float(tau_value)

        record.z_hat = z_prediction  # type: ignore[attr-defined]
        record.s_cls = s_cls  # type: ignore[attr-defined]
        record.s_time = s_time  # type: ignore[attr-defined]
        record.S = combined  # type: ignore[attr-defined]
        record.flag_cls = int(probability_vector.argmax().item() != actual_index)  # type: ignore[attr-defined]
        record.flag_dt = int(tau_hi is not None and s_time > tau_hi)  # type: ignore[attr-defined]


def _score_records(
    model: AnomalyDetectorModel,
    records: Sequence[ContractRecord],
    *,
    category_to_index: Mapping[str, int],
    thresholds: Mapping[str, Any],
    weight_cls: float,
    weight_time: float,
    device: torch.device,
) -> None:
    """Score all records grouped by session."""

    grouped: Dict[tuple[str, str], List[ContractRecord]] = {}
    for record in sorted(
        records, key=lambda item: (item.uid, item.session_id, item.timestamp_utc)
    ):
        key = (record.uid, record.session_id)
        grouped.setdefault(key, []).append(record)

    for session_records in grouped.values():
        _score_session(
            model,
            session_records,
            category_to_index=category_to_index,
            thresholds=thresholds,
            weight_cls=weight_cls,
            weight_time=weight_time,
            device=device,
        )


def _write_scored_csv(
    records: Sequence[ContractRecord], output_path: Path
) -> List[Dict[str, Any]]:
    """Persist scored records to a CSV file with the specified schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SCORE_COLUMNS))
        writer.writeheader()
        for record in records:
            row = {
                "timestamp_utc": record.timestamp_utc.isoformat().replace(
                    "+00:00", "Z"
                ),
                "uid": record.uid,
                "session_id": record.session_id,
                "op_category": record.op_category,
                "z": float(record.z_clipped),
                "z_hat": float(getattr(record, "z_hat", 0.0)),
                "s_cls": float(getattr(record, "s_cls", 0.0)),
                "s_time": float(getattr(record, "s_time", 0.0)),
                "S": float(getattr(record, "S", 0.0)),
                "flag_cls": int(getattr(record, "flag_cls", 0)),
                "flag_dt": int(getattr(record, "flag_dt", 0)),
            }
            writer.writerow(row)
            rows.append(row)
    return rows


def score_dataset(
    *,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    weight_cls: float = DEFAULT_WEIGHT_CLS,
    weight_time: float = DEFAULT_WEIGHT_TIME,
    checkpoint: _Checkpoint | None = None,
) -> ScoreResult:
    """Score an input contract CSV using a trained model checkpoint."""

    if weight_cls < 0 or weight_time < 0:
        raise ValueError("Weights must be non-negative values")

    input_path = Path(input_path)
    model_path = Path(model_path)
    output_path = Path(output_path)

    rows = load_contract_dataframe(input_path)
    checkpoint = checkpoint or _load_checkpoint(model_path)
    clip_value = float(checkpoint.config.get("clip_value", 5.0))

    records = _prepare_records(rows, clip_value=clip_value)
    thresholds_path = model_path.parent / "metrics.json"
    thresholds = _load_thresholds(thresholds_path)

    seed = int(checkpoint.config.get("seed", 0)) if "seed" in checkpoint.config else 0
    set_deterministic_mode(seed)

    category_to_index = {
        str(category): index for index, category in enumerate(checkpoint.encoder_vocab)
    }

    device = _resolve_device()
    model = _build_model(checkpoint).to(device)

    _score_records(
        model,
        records,
        category_to_index=category_to_index,
        thresholds=thresholds,
        weight_cls=weight_cls,
        weight_time=weight_time,
        device=device,
    )

    scored_rows = _write_scored_csv(records, output_path)
    return ScoreResult(rows=scored_rows, seed=seed)


def run_score_command(
    *,
    model: Path,
    input_path: Path,
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Entry point used by the CLI to execute scoring."""

    model = Path(model)
    input_path = Path(input_path)
    output_path = Path(output_path)

    try:
        input_sha = _compute_file_sha256(input_path)
    except FileNotFoundError:
        input_sha = None
    try:
        model_sha = _compute_file_sha256(model)
    except FileNotFoundError:
        model_sha = None

    checkpoint: _Checkpoint | None = None
    seed: int = 0
    try:
        checkpoint = _load_checkpoint(model)
        if "seed" in checkpoint.config:
            seed = int(checkpoint.config.get("seed", 0))
    except Exception as error:
        payload = {
            "model": str(model),
            "input": str(input_path),
            "output": str(output_path),
            "input_sha256": input_sha,
            "model_sha256": model_sha,
            "seed": seed,
        }
        log_event("score_start", payload={"status": "started", **payload})
        log_event(
            "score_error",
            payload={
                "status": "failed",
                "error_code": error.__class__.__name__,
                "message": str(error),
                **payload,
            },
        )
        raise

    payload = {
        "model": str(model),
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": input_sha,
        "model_sha256": model_sha,
        "seed": seed,
    }

    log_event("score_start", payload={"status": "started", **payload})
    try:
        result = score_dataset(
            model_path=model,
            input_path=input_path,
            output_path=output_path,
            weight_cls=DEFAULT_WEIGHT_CLS,
            weight_time=DEFAULT_WEIGHT_TIME,
            checkpoint=checkpoint,
        )
    except Exception as error:
        log_event(
            "score_error",
            payload={
                "status": "failed",
                "error_code": error.__class__.__name__,
                "message": str(error),
                **payload,
            },
        )
        raise

    log_event(
        "score_complete",
        payload={
            "status": "succeeded",
            "rows": len(result.rows),
            **payload,
        },
    )
    return result.rows


__all__ = [
    "SCORE_COLUMNS",
    "score_dataset",
    "ScoreResult",
    "run_score_command",
]
