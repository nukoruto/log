"""Data contract CLI implementation for validate/sessionize/deltify."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .contract import CONTRACT_COLUMNS, REQUIRED_CONTRACT_FIELDS, ensure_required_fields
from .sessionize import (
    EPSILON,
    SESSION_COLUMNS,
    sessionize_contract,
    quantile,
)


DELTIFIED_COLUMNS: list[str] = SESSION_COLUMNS + [
    "robust_z",
    "robust_z_clipped",
    "prev_q25_seconds",
    "prev_q75_seconds",
    "burst_ratio",
]

MAD_SCALE = 1.4826
CLIP_RANGE = (-5.0, 5.0)


class CommandError(RuntimeError):
    """Raised when CLI input validation fails."""

    def __init__(
        self, code: str, message: str, *, hint: str | None = None, exit_code: int = 1
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.exit_code = exit_code


@dataclass(slots=True)
class JsonLogger:
    """Utility emitting JSON structured logs to stdout."""

    command: str
    seed: int | None

    def emit(self, *, event: str, **payload: object) -> None:
        record: dict[str, object] = {
            "event": event,
            "command": self.command,
            "seed": self.seed,
        }
        for key, value in payload.items():
            record[key] = value
        print(json.dumps(record, ensure_ascii=False))

    def log_start(self, details: dict[str, object]) -> None:
        self.emit(event="start", details=details)

    def log_complete(self, details: dict[str, object]) -> None:
        self.emit(event="complete", details=details)

    def log_error(self, *, code: str, message: str, hint: str | None = None) -> None:
        payload: dict[str, object] = {"code": code, "message": message}
        if hint:
            payload["hint"] = hint
        self.emit(event="error", **payload)


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty sequence")
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _mad(values: list[float], median: float) -> float:
    deviations = [abs(value - median) for value in values]
    if not deviations:
        return 0.0
    return _median(deviations)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        prog="ds-contract", description="Data contract processing CLI"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Deterministic seed applied across all subcommands",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    _register_validate(subparsers)
    _register_sessionize(subparsers)
    _register_deltify(subparsers)

    return parser


def _register_validate(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "validate", help="Normalize raw CSV into contract.csv"
    )
    parser.add_argument("input_csv", help="Raw CSV to validate")
    parser.add_argument(
        "--map", dest="mapping", required=True, help="YAML column mapping file"
    )
    parser.add_argument("--out", required=True, help="Contract CSV output path")
    parser.add_argument(
        "--meta",
        help="Metadata JSON output path (default: <out>.meta.json)",
    )
    parser.set_defaults(handler=_handle_validate)


def _register_sessionize(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("sessionize", help="Derive sessions and Δt gaps")
    parser.add_argument("contract_csv", help="Validated contract CSV input")
    parser.add_argument("--out", required=True, help="Sessionized CSV output path")
    parser.add_argument(
        "--meta", required=True, help="Session metadata JSON output path"
    )
    parser.set_defaults(handler=_handle_sessionize)


def _register_deltify(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("deltify", help="Compute robust Δt features")
    parser.add_argument("sessioned_csv", help="Sessionized CSV input")
    parser.add_argument("--out", required=True, help="Deltified CSV output path")
    parser.add_argument("--meta", required=True, help="Δt metadata JSON output path")
    parser.set_defaults(handler=_handle_deltify)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0 and argv and any(opt in argv for opt in {"--help", "-h"}):
            return 0
        raise

    if getattr(args, "handler", None) is None:
        parser.print_help()
        return 0

    logger = JsonLogger(command=args.command, seed=getattr(args, "seed", None))

    if getattr(args, "seed", None) is None:
        logger.log_error(
            code="MISSING_SEED",
            message="--seed is required for deterministic processing.",
            hint="Invoke the command with an explicit --seed integer value.",
        )
        return 1

    _set_global_seed(args.seed)

    try:
        result = args.handler(args, logger)
    except CommandError as exc:
        logger.log_error(code=exc.code, message=exc.message, hint=exc.hint)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.log_error(code="UNEXPECTED", message=str(exc))
        return 1
    else:
        logger.log_complete(result)
        return 0


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        return


def _default_meta_path(output_path: Path) -> Path:
    stem = output_path.stem or output_path.name or "contract"
    return output_path.with_name(f"{stem}.meta.json")


def _handle_validate(args: argparse.Namespace, logger: JsonLogger) -> dict[str, object]:
    input_path = Path(args.input_csv)
    mapping_path = Path(args.mapping)
    output_path = Path(args.out)
    meta_arg = getattr(args, "meta", None)
    if meta_arg:
        meta_path = Path(meta_arg)
    else:
        meta_path = _default_meta_path(output_path)

    logger.log_start(
        {
            "input": str(input_path),
            "mapping": str(mapping_path),
            "output": str(output_path),
            "meta": str(meta_path),
        }
    )

    if not input_path.exists():
        raise CommandError(
            "INPUT_NOT_FOUND",
            f"Input CSV '{input_path}' does not exist.",
            hint="Provide a valid raw CSV file for validation.",
        )

    if not mapping_path.exists():
        raise CommandError(
            "MAPPING_NOT_FOUND",
            f"Mapping YAML '{mapping_path}' does not exist.",
            hint="Pass --map with a column mapping YAML file.",
        )

    mapping = _load_mapping(mapping_path)
    rows = _read_and_normalize_contract(input_path, mapping)
    _write_csv(output_path, CONTRACT_COLUMNS, rows)

    meta = {
        "seed": args.seed,
        "input_sha256": _compute_sha256(input_path),
        "row_count": len(rows),
        "mapping": mapping,
        "required_columns": sorted(REQUIRED_CONTRACT_FIELDS),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "row_count": len(rows),
        "input_sha256": meta["input_sha256"],
    }


def _handle_sessionize(
    args: argparse.Namespace, logger: JsonLogger
) -> dict[str, object]:
    input_path = Path(args.contract_csv)
    output_path = Path(args.out)
    meta_path = Path(args.meta)

    logger.log_start(
        {"input": str(input_path), "output": str(output_path), "meta": str(meta_path)}
    )

    rows = _load_contract_rows(input_path)
    if not rows:
        raise CommandError(
            "EMPTY_CONTRACT",
            "Contract CSV contains no rows to sessionize.",
            hint="Run validate on a non-empty dataset before sessionize.",
        )

    result = sessionize_contract(rows)
    sessionized = result.rows

    _write_csv(output_path, SESSION_COLUMNS, sessionized)

    meta = {
        "seed": args.seed,
        "input_sha256": _compute_sha256(input_path),
        "method": result.method,
        "threshold_seconds": result.threshold_seconds,
        "log_threshold": result.log_threshold,
        "histogram": result.histogram,
        "bins": result.bins,
        "within_class_variance_ratio": result.within_class_variance_ratio,
        "epsilon": EPSILON,
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "row_count": len(sessionized),
        "threshold_seconds": result.threshold_seconds,
        "method": result.method,
        "input_sha256": meta["input_sha256"],
    }


def _handle_deltify(args: argparse.Namespace, logger: JsonLogger) -> dict[str, object]:
    input_path = Path(args.sessioned_csv)
    output_path = Path(args.out)
    meta_path = Path(args.meta)

    logger.log_start(
        {"input": str(input_path), "output": str(output_path), "meta": str(meta_path)}
    )

    rows = _load_sessioned_rows(input_path)
    if not rows:
        raise CommandError(
            "EMPTY_SESSIONED",
            "Sessioned CSV contains no rows for Δt feature extraction.",
            hint="Run sessionize before deltify on a populated dataset.",
        )

    deltified, meta = _compute_robust_features(rows)
    meta.update(
        {
            "seed": args.seed,
            "input_sha256": _compute_sha256(input_path),
            "clip_range": list(CLIP_RANGE),
        }
    )
    _write_csv(output_path, DELTIFIED_COLUMNS, deltified)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "row_count": len(deltified),
        "fallback_users": meta.get("fallback_users", []),
        "input_sha256": meta["input_sha256"],
    }


def _load_mapping(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = _parse_simple_yaml(text)

    mapping = loaded
    if not isinstance(mapping, dict):
        raise CommandError(
            "INVALID_MAPPING",
            "Mapping YAML must define a dictionary from contract columns to input columns.",
            hint="Ensure the YAML has key: value pairs for each contract column.",
        )

    normalized: dict[str, str] = {}
    for key, value in mapping.items():
        if key not in CONTRACT_COLUMNS:
            raise CommandError(
                "UNKNOWN_MAPPING_KEY",
                f"Unsupported contract column '{key}' in mapping file.",
                hint="Use only columns defined in the 9-column contract schema.",
            )
        if not isinstance(value, str):
            raise CommandError(
                "INVALID_MAPPING_VALUE",
                f"Mapping for '{key}' must be a string column name.",
            )
        normalized[key] = value

    missing = ensure_required_fields(normalized.keys())
    required_missing = frozenset(field for field in missing if field != "ip")
    if required_missing:
        raise CommandError(
            "MISSING_MAPPING",
            f"Mapping file lacks required contract columns: {sorted(required_missing)}.",
            hint="Add source column mappings for all required contract fields.",
        )
    return normalized


def _parse_simple_yaml(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise CommandError(
                "INVALID_MAPPING_LINE",
                f"Unable to parse mapping line '{raw_line}'.",
                hint="Use 'key: value' pairs in the mapping file.",
            )
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        mapping[key] = value
    return mapping


def _read_and_normalize_contract(
    input_path: Path, mapping: dict[str, str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [
            source
            for source in mapping.values()
            if source not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise CommandError(
                "MISSING_INPUT_COLUMNS",
                f"Input CSV is missing mapped columns: {missing_columns}",
                hint="Verify the raw CSV headers and the mapping YAML entries.",
            )

        for index, raw_row in enumerate(reader, start=1):
            contract_row: dict[str, str] = {}
            for column in CONTRACT_COLUMNS:
                source_column = mapping.get(column)
                value = raw_row.get(source_column, "") if source_column else ""
                if column in REQUIRED_CONTRACT_FIELDS and column != "ip" and not value:
                    raise CommandError(
                        "MISSING_REQUIRED_VALUE",
                        f"Row {index} is missing required column '{source_column}'.",
                        hint="Ensure required columns are populated in the raw CSV.",
                    )
                if column == "timestamp_utc":
                    contract_row[column] = _normalize_timestamp(value)
                else:
                    contract_row[column] = value
            rows.append(contract_row)

    rows.sort(key=lambda item: item["timestamp_utc"])
    return rows


def _normalize_timestamp(value: str) -> str:
    try:
        if value.isdigit():
            timestamp_ms = int(value)
            if len(value) > 10:
                seconds = timestamp_ms / 1000
            else:
                seconds = float(timestamp_ms)
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
            return dt.isoformat()
        try:
            seconds = float(value)
        except ValueError:
            pass
        else:
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
            return dt.isoformat()

        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            raise ValueError
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError as exc:
        raise CommandError(
            "INVALID_TIMESTAMP",
            f"Timestamp '{value}' is not a valid UTC value.",
            hint="Provide ISO8601 (with timezone) or epoch seconds/milliseconds.",
        ) from exc


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]
) -> None:
    field_list = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=field_list)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise CommandError(
            "CONTRACT_NOT_FOUND",
            f"Contract CSV '{path}' is missing.",
            hint="Run validate before sessionize.",
        )

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CONTRACT_COLUMNS):
            raise CommandError(
                "INVALID_CONTRACT_HEADER",
                "Contract CSV must match the 9-column contract schema.",
            )
        rows = [dict(row) for row in reader]

    rows.sort(key=lambda item: item["timestamp_utc"])
    return rows


def _load_sessioned_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise CommandError(
            "SESSIONED_NOT_FOUND",
            f"Sessioned CSV '{path}' is missing.",
            hint="Run sessionize before deltify.",
        )

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = SESSION_COLUMNS
        if reader.fieldnames != expected:
            raise CommandError(
                "INVALID_SESSIONED_HEADER",
                "Sessioned CSV must include contract columns plus delta_t/log_delta_t.",
            )
        rows = [dict(row) for row in reader]

    rows.sort(key=lambda item: item["timestamp_utc"])
    return rows


def _compute_robust_features(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    log_values_per_user: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        uid = row["uid"]
        delta_seconds = float(row["delta_t_seconds"])
        log_value = math.log(delta_seconds + EPSILON)
        log_values_per_user[uid].append(log_value)

    all_log_values = [
        value for values in log_values_per_user.values() for value in values
    ]
    global_median = float(_median(all_log_values))
    global_mad = float(_mad(all_log_values, global_median))
    if global_mad == 0:
        global_mad = 1e-9

    user_stats: dict[str, tuple[float, float]] = {}
    fallback_users: list[str] = []
    for uid, values in log_values_per_user.items():
        if len(values) >= 5:
            median = float(_median(values))
            mad = float(_mad(values, median))
        else:
            median = global_median
            mad = global_mad
            fallback_users.append(uid)
        if mad == 0:
            mad = global_mad
        user_stats[uid] = (median, mad)

    enriched_rows: list[dict[str, str]] = []
    previous_deltas: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        uid = row["uid"]
        delta_seconds = float(row["delta_t_seconds"])
        log_delta = math.log(delta_seconds + EPSILON)
        median, mad = user_stats[uid]
        denominator = MAD_SCALE * mad if mad else MAD_SCALE * global_mad
        z_score = (log_delta - median) / denominator
        clipped = max(CLIP_RANGE[0], min(CLIP_RANGE[1], z_score))

        history = previous_deltas[uid]
        if history:
            q25 = float(quantile(history, 0.25))
            q75 = float(quantile(history, 0.75))
            prev_delta = history[-1]
        else:
            q25 = delta_seconds
            q75 = delta_seconds
            prev_delta = delta_seconds

        burst = (prev_delta + EPSILON) / (delta_seconds + EPSILON)

        enriched = dict(row)
        enriched["robust_z"] = f"{z_score:.6f}"
        enriched["robust_z_clipped"] = f"{clipped:.6f}"
        enriched["prev_q25_seconds"] = f"{q25:.6f}"
        enriched["prev_q75_seconds"] = f"{q75:.6f}"
        enriched["burst_ratio"] = f"{burst:.6f}"
        enriched_rows.append(enriched)

        history.append(delta_seconds)
        if len(history) > 5:
            history.pop(0)

    meta = {
        "epsilon": EPSILON,
        "mad_scale": MAD_SCALE,
        "clip_range": list(CLIP_RANGE),
        "global_median": global_median,
        "global_mad": global_mad,
        "group": "uid",
        "fallback_users": fallback_users,
        "user_stats": {
            uid: {"median": stats[0], "mad": stats[1]}
            for uid, stats in user_stats.items()
        },
    }

    return enriched_rows, meta


if __name__ == "__main__":  # pragma: no cover - CLI execution guard
    raise SystemExit(main())
