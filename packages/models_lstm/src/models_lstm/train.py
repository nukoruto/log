"""Training utilities for the models-lstm package."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy  # type: ignore[import-not-found]

import torch  # type: ignore[import-not-found]
from torch import Tensor, nn  # type: ignore[attr-defined]
from torch.cuda.amp import GradScaler, autocast  # type: ignore[attr-defined, import-not-found]
from torch.utils.data import DataLoader, Dataset  # type: ignore[attr-defined, import-not-found]

from .data import (
    SequenceExample,
    build_sequence_examples,
    load_contract_dataframe,
    OpCategoryEncoder,
)
from .metrics import build_category_thresholds, save_metrics
from .model import LSTMEventPredictor


@dataclass
class TrainingConfig:
    """Configuration values required for training."""

    normal_path: Path
    val_path: Path
    output_dir: Path
    seed: int
    batch_size: int = 256
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    lambda_huber: float = 1.0
    huber_delta: float = 1.0
    epochs: int = 50
    patience: int = 5
    clip_value: float = 5.0
    use_amp: bool = True


class SessionSequenceDataset(Dataset):
    """Dataset that converts session sequences into training examples."""

    def __init__(self, sequences: Sequence[SequenceExample]) -> None:
        examples: List[Tuple[List[int], List[float], List[int], List[float]]] = []
        for sequence in sequences:
            length = len(sequence.op_indices)
            if length < 2:
                continue
            op_inputs = [int(index) for index in sequence.op_indices[:-1]]
            z_inputs = [float(value) for value in sequence.z_clipped[:-1]]
            target_ops = [int(index) for index in sequence.op_indices[1:]]
            target_z = [float(value) for value in sequence.z_clipped[1:]]
            examples.append((op_inputs, z_inputs, target_ops, target_z))
        if not examples:
            raise ValueError(
                "Dataset is empty. Ensure sequences contain at least two events."
            )
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(
        self, index: int
    ) -> Tuple[List[int], List[float], List[int], List[float]]:
        return self._examples[index]


def _collate_examples(
    batch: Sequence[Tuple[List[int], List[float], List[int], List[float]]],
    *,
    padding_index: int,
) -> Dict[str, Tensor]:
    batch_size = len(batch)
    max_length = max(len(example[0]) for example in batch)

    op_inputs = torch.full((batch_size, max_length), padding_index, dtype=torch.long)
    z_inputs = torch.zeros((batch_size, max_length), dtype=torch.float32)
    target_ops = torch.zeros((batch_size, max_length), dtype=torch.long)
    target_z = torch.zeros((batch_size, max_length), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    for row, (ops, z_values, ops_target, z_target) in enumerate(batch):
        length = len(ops)
        if length == 0:
            continue
        op_inputs[row, :length] = torch.tensor(ops, dtype=torch.long) + 1
        z_inputs[row, :length] = torch.tensor(z_values, dtype=torch.float32)
        target_ops[row, :length] = torch.tensor(ops_target, dtype=torch.long)
        target_z[row, :length] = torch.tensor(z_target, dtype=torch.float32)
        mask[row, :length] = True

    return {
        "op_inputs": op_inputs,
        "z_inputs": z_inputs,
        "target_ops": target_ops,
        "target_z": target_z,
        "mask": mask,
    }


class AnomalyDetectorModel(nn.Module):
    """Composite model including embeddings and the LSTM predictor."""

    def __init__(
        self,
        *,
        num_categories: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_categories + 1,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.predictor = LSTMEventPredictor(
            input_dim=embed_dim + 1,
            hidden_dim=hidden_dim,
            num_op_categories=num_categories,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, op_indices: Tensor, z_inputs: Tensor) -> Tuple[Tensor, Tensor]:
        embeddings = self.embedding(op_indices)
        features = torch.cat([embeddings, z_inputs.unsqueeze(-1)], dim=-1)
        return self.predictor(features)


def set_deterministic_mode(seed: int) -> None:
    """Seed all major RNGs for deterministic execution."""

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:  # pragma: no cover - deterministic mode may be unavailable
        pass


def _resolve_device() -> torch.device:
    mode = os.environ.get("GPU_MODE", "cpu").lower()
    if mode == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    target_index: int | None = None
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index).lower()
        if mode in name:
            target_index = index
            break

    if target_index is None:
        if torch.cuda.is_available():
            target_index = 0
        else:
            return torch.device("cpu")
    return torch.device(f"cuda:{target_index}")


def _compute_file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reencode_sequences(
    sequences: Sequence[SequenceExample],
    *,
    source_encoder: OpCategoryEncoder,
    target_encoder: OpCategoryEncoder,
) -> List[SequenceExample]:
    reencoded: List[SequenceExample] = []
    for sequence in sequences:
        categories = source_encoder.inverse_transform(sequence.op_indices)
        op_indices = target_encoder.transform(categories)
        reencoded.append(
            SequenceExample(
                uid=sequence.uid,
                session_id=sequence.session_id,
                timestamps=sequence.timestamps,
                op_indices=op_indices,
                delta_seconds=sequence.delta_seconds,
                z_clipped=sequence.z_clipped,
            )
        )
    return reencoded


def _create_dataloader(
    sequences: Sequence[SequenceExample],
    *,
    batch_size: int,
    padding_index: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Dict[str, Tensor]]:
    dataset = SessionSequenceDataset(sequences)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda batch: _collate_examples(batch, padding_index=padding_index),
        generator=generator if shuffle else None,
    )


def _classification_loss(op_probs: Tensor, targets: Tensor) -> Tensor:
    probabilities = torch.clamp(op_probs, min=1e-9, max=1.0)
    gathered = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -torch.log(gathered).mean()


def _huber_loss(predictions: Tensor, targets: Tensor, delta: float) -> Tensor:
    loss_fn = nn.SmoothL1Loss(beta=delta)
    return loss_fn(predictions, targets)


def _evaluate(
    model: AnomalyDetectorModel,
    dataloader: DataLoader[Dict[str, Tensor]],
    *,
    device: torch.device,
    lambda_huber: float,
    huber_delta: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_huber = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            mask = batch["mask"].to(device)
            if not mask.any():
                continue
            op_inputs = batch["op_inputs"].to(device)
            z_inputs = batch["z_inputs"].to(device)
            target_ops = batch["target_ops"].to(device)
            target_z = batch["target_z"].to(device)

            op_probs, z_hat = model(op_inputs, z_inputs)
            mask_indices = mask.view(-1)
            probs_flat = op_probs.view(-1, op_probs.size(-1))[mask_indices]
            targets_flat = target_ops.view(-1)[mask_indices]
            z_hat_flat = z_hat.view(-1)[mask_indices]
            target_z_flat = target_z.view(-1)[mask_indices]

            ce = _classification_loss(probs_flat, targets_flat)
            huber = _huber_loss(z_hat_flat, target_z_flat, huber_delta)
            loss = ce + lambda_huber * huber

            total_steps += probs_flat.size(0)
            total_ce += ce.item() * probs_flat.size(0)
            total_huber += huber.item() * probs_flat.size(0)
            total_loss += loss.item() * probs_flat.size(0)

    if total_steps == 0:
        raise ValueError("Validation set contains no usable sequences.")

    return (
        total_loss / total_steps,
        {
            "ce": total_ce / total_steps,
            "huber": total_huber / total_steps,
        },
    )


def _micro_f1(targets: Sequence[int], predictions: Sequence[int]) -> float:
    """Compute micro-averaged F1 for multi-class predictions."""

    total = len(targets)
    if total == 0:
        raise ValueError("targets must not be empty")
    true_positive = sum(
        1 for target, pred in zip(targets, predictions) if target == pred
    )
    if true_positive == 0:
        return 0.0
    precision = true_positive / total
    recall = precision
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Calculate ROC-AUC for binary labels using average rank handling ties."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    current_rank = 1.0
    i = 0
    while i < len(order):
        j = i
        value = scores[order[i]]
        while j < len(order) and scores[order[j]] == value:
            j += 1
        average_rank = (2 * current_rank + (j - i) - 1) / 2
        for position in range(i, j):
            ranks[order[position]] = average_rank
        current_rank += j - i
        i = j

    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def _binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute average precision (area under PR curve) for binary labels."""

    positives = sum(labels)
    if positives == 0:
        return math.nan

    sorted_pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0.0
    fp = 0.0
    last_recall = 0.0
    area = 0.0
    for score, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += precision * max(0.0, recall - last_recall)
        last_recall = recall
    return float(area)


def _average_over_classes(
    targets: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
) -> Optional[float]:
    """Apply a binary metric across classes and average the valid values."""

    if not probabilities:
        return None
    num_classes = len(probabilities[0])
    values: List[float] = []
    for class_index in range(num_classes):
        labels = [1 if target == class_index else 0 for target in targets]
        scores = [prob[class_index] for prob in probabilities]
        value = metric_fn(labels, scores)
        if not math.isnan(value):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _build_thresholds(
    residuals_by_category: Dict[str, List[float]],
    categories: Sequence[str],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Proxy wrapper that emits linear-quantile thresholds for residuals."""

    return build_category_thresholds(residuals_by_category, categories, alpha=alpha)


def _estimate_detection_delay(
    model: AnomalyDetectorModel,
    sequences: Sequence[SequenceExample],
    *,
    device: torch.device,
    thresholds: Dict[str, Any],
    categories: Sequence[str],
) -> float:
    """Estimate detection delay as the mean time to first trigger per sequence."""

    per_category = thresholds.get("per_category", {})
    detection_times: List[float] = []
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            length = len(sequence.op_indices)
            if length < 2:
                continue

            op_tensor = torch.tensor(
                [index + 1 for index in sequence.op_indices[:-1]],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            z_tensor = torch.tensor(
                list(sequence.z_clipped[:-1]),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            op_probs, z_hat = model(op_tensor, z_tensor)
            probs_cpu = op_probs.squeeze(0).detach().cpu()
            z_hat_cpu = z_hat.squeeze(0).detach().cpu()
            predicted_classes = probs_cpu.argmax(dim=1)

            cumulative_time = 0.0
            detection_time = 0.0
            detected = False

            for step in range(length - 1):
                cumulative_time += float(sequence.delta_seconds[step + 1])
                target_index = int(sequence.op_indices[step + 1])
                category = categories[target_index]
                tau_info = per_category.get(category)
                tau_hi: Optional[float] = None
                if isinstance(tau_info, dict):
                    tau_value = tau_info.get("tau_hi")
                    if tau_value is not None:
                        tau_hi = float(tau_value)

                residual = abs(
                    float(z_hat_cpu[step].item()) - float(sequence.z_clipped[step + 1])
                )
                classification_mismatch = (
                    int(predicted_classes[step].item()) != target_index
                )

                triggered = False
                if tau_hi is not None and residual > tau_hi:
                    triggered = True
                if not triggered and classification_mismatch:
                    triggered = True

                if triggered:
                    detection_time = cumulative_time
                    detected = True
                    break

            if not detected:
                detection_time = cumulative_time

            detection_times.append(detection_time)

    if not detection_times:
        return 0.0
    return float(sum(detection_times) / len(detection_times))


def _compute_validation_metrics(
    model: AnomalyDetectorModel,
    dataloader: DataLoader[Dict[str, Tensor]],
    *,
    device: torch.device,
    encoder: OpCategoryEncoder,
    sequences: Sequence[SequenceExample],
) -> Dict[str, Any]:
    """Evaluate the best model checkpoint on the validation split."""

    categories = list(encoder.inverse_transform(range(encoder.vocab_size)))
    target_indices: List[int] = []
    probability_vectors: List[List[float]] = []
    residuals_by_category: DefaultDict[str, List[float]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            mask = batch["mask"].to(device)
            if not mask.any():
                continue
            op_inputs = batch["op_inputs"].to(device)
            z_inputs = batch["z_inputs"].to(device)
            target_ops = batch["target_ops"].to(device)
            target_z = batch["target_z"].to(device)

            op_probs, z_hat = model(op_inputs, z_inputs)
            mask_indices = mask.view(-1)
            probs_flat = op_probs.view(-1, op_probs.size(-1))[mask_indices]
            targets_flat = target_ops.view(-1)[mask_indices]
            z_hat_flat = z_hat.view(-1)[mask_indices]
            target_z_flat = target_z.view(-1)[mask_indices]

            probs_cpu = probs_flat.detach().cpu()
            targets_cpu = targets_flat.detach().cpu()
            z_hat_cpu = z_hat_flat.detach().cpu()
            target_z_cpu = target_z_flat.detach().cpu()

            for index in range(probs_cpu.size(0)):
                target_index = int(targets_cpu[index].item())
                probability_vectors.append(
                    [float(value) for value in probs_cpu[index].tolist()]
                )
                target_indices.append(target_index)
                residual = abs(
                    float(z_hat_cpu[index].item()) - float(target_z_cpu[index].item())
                )
                category = categories[target_index]
                residuals_by_category[category].append(residual)

    if not target_indices:
        raise ValueError("Validation set contains no usable sequences.")

    predictions = [
        max(range(len(prob_vector)), key=lambda idx: prob_vector[idx])
        for prob_vector in probability_vectors
    ]

    f1_score = _micro_f1(target_indices, predictions)
    pr_auc = _average_over_classes(
        target_indices, probability_vectors, _binary_average_precision
    )
    roc_auc = _average_over_classes(
        target_indices, probability_vectors, _binary_roc_auc
    )
    thresholds = _build_thresholds(dict(residuals_by_category), categories)
    detection_delay = _estimate_detection_delay(
        model,
        sequences,
        device=device,
        thresholds=thresholds,
        categories=categories,
    )

    return {
        "f1": f1_score,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "detection_delay": detection_delay,
        "thresholds": thresholds,
    }


def _train_one_epoch(
    model: AnomalyDetectorModel,
    dataloader: DataLoader[Dict[str, Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    lambda_huber: float,
    huber_delta: float,
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_huber = 0.0
    total_steps = 0

    for batch in dataloader:
        mask = batch["mask"].to(device)
        if not mask.any():
            continue
        op_inputs = batch["op_inputs"].to(device)
        z_inputs = batch["z_inputs"].to(device)
        target_ops = batch["target_ops"].to(device)
        target_z = batch["target_z"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            op_probs, z_hat = model(op_inputs, z_inputs)
            mask_indices = mask.view(-1)
            probs_flat = op_probs.view(-1, op_probs.size(-1))[mask_indices]
            targets_flat = target_ops.view(-1)[mask_indices]
            z_hat_flat = z_hat.view(-1)[mask_indices]
            target_z_flat = target_z.view(-1)[mask_indices]

            ce = _classification_loss(probs_flat, targets_flat)
            huber = _huber_loss(z_hat_flat, target_z_flat, huber_delta)
            loss = ce + lambda_huber * huber

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_steps += probs_flat.size(0)
        total_ce += ce.item() * probs_flat.size(0)
        total_huber += huber.item() * probs_flat.size(0)
        total_loss += loss.item() * probs_flat.size(0)

    if total_steps == 0:
        raise ValueError("Training set contains no usable sequences.")

    return (
        total_loss / total_steps,
        {
            "ce": total_ce / total_steps,
            "huber": total_huber / total_steps,
        },
    )


def train_model(config: TrainingConfig) -> Dict[str, Any]:
    """Execute the training loop and persist metrics and checkpoints."""

    if config.epochs <= 0 or config.epochs > 50:
        raise ValueError("epochs must be in the range [1, 50]")
    if config.patience <= 0:
        raise ValueError("patience must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic_mode(config.seed)
    device = _resolve_device()

    train_records = load_contract_dataframe(config.normal_path)
    train_sequences, encoder, train_stats = build_sequence_examples(
        train_records, clip_value=config.clip_value
    )

    val_records = load_contract_dataframe(config.val_path)
    val_sequences_temp, val_encoder, val_stats = build_sequence_examples(
        val_records, clip_value=config.clip_value
    )
    val_sequences = _reencode_sequences(
        val_sequences_temp,
        source_encoder=val_encoder,
        target_encoder=encoder,
    )

    padding_index = 0
    train_loader = _create_dataloader(
        train_sequences,
        batch_size=config.batch_size,
        padding_index=padding_index,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = _create_dataloader(
        val_sequences,
        batch_size=config.batch_size,
        padding_index=padding_index,
        shuffle=False,
        seed=config.seed,
    )

    model = AnomalyDetectorModel(
        num_categories=encoder.vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")

    best_state: Dict[str, Tensor] | None = None
    best_epoch = -1
    best_val_loss = math.inf
    epochs_without_improvement = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_parts = _train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            lambda_huber=config.lambda_huber,
            huber_delta=config.huber_delta,
            use_amp=config.use_amp and device.type == "cuda",
        )

        val_loss, val_parts = _evaluate(
            model,
            val_loader,
            device=device,
            lambda_huber=config.lambda_huber,
            huber_delta=config.huber_delta,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_ce": float(train_parts["ce"]),
                "train_huber": float(train_parts["huber"]),
                "val_loss": float(val_loss),
                "val_ce": float(val_parts["ce"]),
                "val_huber": float(val_parts["huber"]),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": best_state,
                    "encoder_vocab": encoder.inverse_transform(
                        range(encoder.vocab_size)
                    ),
                    "config": asdict(config),
                },
                output_dir / "best.ckpt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    validation_metrics = _compute_validation_metrics(
        model,
        val_loader,
        device=device,
        encoder=encoder,
        sequences=val_sequences,
    )

    normal_sha = _compute_file_sha256(config.normal_path)
    val_sha = _compute_file_sha256(config.val_path)

    metrics: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "loss": best_val_loss,
        "history": history,
        "normal_sha256": normal_sha,
        "val_sha256": val_sha,
        "seed": config.seed,
        "device": str(device),
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    metrics.update(validation_metrics)

    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics_path, metrics)

    return metrics


def log_event(event: str, *, payload: Dict[str, Any]) -> None:
    message = {"event": event, **payload}
    print(json.dumps(message), flush=True)


def run_train_command(
    *,
    normal: Path,
    val: Path,
    out: Path,
    seed: int,
    batch_size: int = 256,
    epochs: int = 50,
    patience: int = 5,
    learning_rate: float = 1e-3,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.1,
    lambda_huber: float = 1.0,
    huber_delta: float = 1.0,
) -> Dict[str, Any]:
    """Entry point used by the CLI to execute the training routine."""

    config = TrainingConfig(
        normal_path=normal,
        val_path=val,
        output_dir=out,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lambda_huber=lambda_huber,
        huber_delta=huber_delta,
    )
    try:
        normal_sha = _compute_file_sha256(config.normal_path)
    except FileNotFoundError:
        normal_sha = None
    try:
        val_sha = _compute_file_sha256(config.val_path)
    except FileNotFoundError:
        val_sha = None

    shared_payload = {
        "normal": str(config.normal_path),
        "val": str(config.val_path),
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "normal_sha256": normal_sha,
        "val_sha256": val_sha,
        "input_sha256": {
            "normal": normal_sha,
            "val": val_sha,
        },
    }

    log_event(
        "train_start",
        payload={"status": "started", **shared_payload},
    )

    try:
        metrics = train_model(config)
    except Exception as error:
        log_event(
            "train_error",
            payload={
                "status": "failed",
                "error_code": error.__class__.__name__,
                "message": str(error),
                **shared_payload,
            },
        )
        raise

    log_event(
        "train_complete",
        payload={
            "status": "succeeded",
            "metrics_path": str(config.output_dir / "metrics.json"),
            "best_epoch": metrics["best_epoch"],
            **shared_payload,
        },
    )
    return metrics


__all__ = ["TrainingConfig", "train_model", "run_train_command", "log_event"]
