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
from typing import Iterable, Sequence, cast

from .contract import CONTRACT_COLUMNS, REQUIRED_CONTRACT_FIELDS, ensure_required_fields


SESSION_COLUMNS: list[str] = list(CONTRACT_COLUMNS) + [
    "delta_t_seconds",
    "log_delta_t",
]

DELTIFIED_COLUMNS: list[str] = SESSION_COLUMNS + [
    "robust_z",
    "robust_z_clipped",
    "prev_q25_seconds",
    "prev_q75_seconds",
    "burst_ratio",
]

EPSILON = 1e-3
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


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty sequence")
    sorted_values = sorted(values)
    return _quantile_sorted(sorted_values, q)


def _quantile_sorted(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty sequence")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value * (1 - weight) + upper_value * weight


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


def _histogram(
    values: list[float], bins: int, min_value: float, max_value: float
) -> tuple[list[int], list[float]]:
    if bins <= 0:
        bins = 1
    if math.isclose(max_value, min_value):
        return [len(values)], [min_value, max_value]
    width = (max_value - min_value) / bins
    if width <= 0:
        width = 1.0
    edges = [min_value + i * width for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for value in values:
        if value >= max_value:
            index = bins - 1
        else:
            index = int((value - min_value) / width)
            index = max(0, min(index, bins - 1))
        counts[index] += 1
    return counts, edges


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

    threshold_seconds, method, histogram_info, ratio = _select_session_threshold(rows)
    sessionized = _assign_sessions(rows, threshold_seconds)

    _write_csv(output_path, SESSION_COLUMNS, sessionized)

    meta = {
        "seed": args.seed,
        "input_sha256": _compute_sha256(input_path),
        "method": method,
        "threshold_seconds": threshold_seconds,
        "log_threshold": math.log(threshold_seconds + EPSILON),
        "histogram": histogram_info,
        "within_class_variance_ratio": ratio,
        "epsilon": EPSILON,
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "row_count": len(sessionized),
        "threshold_seconds": threshold_seconds,
        "method": method,
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
    if missing:
        raise CommandError(
            "MISSING_MAPPING",
            f"Mapping file lacks required contract columns: {sorted(missing)}.",
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
                if column in REQUIRED_CONTRACT_FIELDS and not value:
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


def _select_session_threshold(
    rows: list[dict[str, str]],
) -> tuple[float, str, dict[str, object], float]:
    delta_seconds: list[float] = []
    last_timestamp_per_uid: dict[str, datetime] = {}
    timestamps_per_uid: dict[str, list[datetime]] = defaultdict(list)

    for row in rows:
        uid = row["uid"]
        timestamp = datetime.fromisoformat(row["timestamp_utc"])
        previous = last_timestamp_per_uid.get(uid)
        if previous is None:
            delta_seconds.append(0.0)
        else:
            delta_seconds.append(max((timestamp - previous).total_seconds(), 0.0))
        last_timestamp_per_uid[uid] = timestamp
        timestamps_per_uid[uid].append(timestamp)

    positive_deltas = [value for value in delta_seconds[1:] if value > 0]
    if not positive_deltas:
        threshold = 3600.0
        histogram = cast(
            dict[str, object],
            {"bin_edges": [], "counts": [], "fd_width": None, "bin_count": 0},
        )
        return threshold, "elbow", histogram, 1.0

    log_values = [math.log(value + EPSILON) for value in positive_deltas]
    min_log = min(log_values)
    max_log = max(log_values)

    if math.isclose(max_log, min_log, rel_tol=1e-12, abs_tol=1e-12):
        histogram = cast(
            dict[str, object],
            {
                "bin_edges": [min_log, max_log],
                "counts": [len(log_values)],
                "fd_width": None,
                "bin_count": 1,
            },
        )
        return math.exp(max_log), "elbow", histogram, 1.0

    q75 = _quantile(log_values, 0.75)
    q25 = _quantile(log_values, 0.25)
    iqr = q75 - q25
    count = len(log_values)
    width = 0.0
    if count > 0:
        width = 2 * iqr / (count ** (1 / 3)) if count > 0 else 0.0
    if width <= 0 or not math.isfinite(width):
        span = max_log - min_log
        width = span / 64 if span > 0 else 1.0

    span = max_log - min_log
    bin_count = int(math.ceil(span / width)) if width > 0 else 32
    bin_count = max(32, min(256, bin_count))

    counts, edges = _histogram(log_values, bin_count, min_log, max_log)
    total = sum(counts)
    probabilities = [count / total if total else 0.0 for count in counts]
    bin_centers = [
        (edges[index] + edges[index + 1]) / 2 for index in range(len(counts))
    ]

    omega: list[float] = []
    mu: list[float] = []
    running_prob = 0.0
    running_mu = 0.0
    for prob, center in zip(probabilities, bin_centers):
        running_prob += prob
        running_mu += prob * center
        omega.append(running_prob)
        mu.append(running_mu)

    mu_total = mu[-1] if mu else 0.0
    sigma_between: list[float] = []
    for idx, w in enumerate(omega):
        if w <= 0 or w >= 1:
            sigma_between.append(0.0)
            continue
        numerator = (mu_total * w - mu[idx]) ** 2
        denominator = w * (1 - w)
        sigma_between.append(numerator / denominator if denominator else 0.0)

    otsu_index = max(range(len(sigma_between)), key=lambda i: sigma_between[i])
    otsu_threshold_log = bin_centers[otsu_index]

    w0 = omega[otsu_index]
    w1 = 1.0 - w0
    variance_total = sum(
        prob * (center - mu_total) ** 2
        for prob, center in zip(probabilities, bin_centers)
    )
    if w0 > 0:
        mu0 = mu[otsu_index] / w0
        var0 = (
            sum(
                probabilities[i] * (bin_centers[i] - mu0) ** 2
                for i in range(otsu_index + 1)
            )
            / w0
        )
    else:
        var0 = 0.0
    if w1 > 0:
        mu1 = (mu_total - mu[otsu_index]) / w1
        var1 = (
            sum(
                probabilities[i] * (bin_centers[i] - mu1) ** 2
                for i in range(otsu_index + 1, len(bin_centers))
            )
            / w1
        )
    else:
        var1 = 0.0
    if variance_total == 0:
        ratio = 1.0
    else:
        ratio = (w0 * var0 + w1 * var1) / variance_total

    histogram = cast(
        dict[str, object],
        {
            "bin_edges": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
            "fd_width": float(width),
            "bin_count": bin_count,
        },
    )

    if ratio < 0.9:
        method = "otsu"
        threshold = math.exp(otsu_threshold_log)
        return threshold, method, histogram, ratio

    method = "elbow"
    threshold = _elbow_threshold(log_values, timestamps_per_uid)
    return threshold, method, histogram, ratio


def _elbow_threshold(
    log_values: list[float], timestamps_per_uid: dict[str, list[datetime]]
) -> float:
    sorted_values = sorted(log_values)
    unique_values = sorted(set(sorted_values))
    if len(unique_values) == 1:
        return math.exp(unique_values[0])

    thresholds = unique_values
    min_val = thresholds[0]
    max_val = thresholds[-1]

    session_counts = []
    for tau in thresholds:
        threshold_seconds = math.exp(tau)
        count = _count_sessions_for_threshold(timestamps_per_uid, threshold_seconds)
        session_counts.append(count)

    range_val = max_val - min_val
    norm_x = [((tau - min_val) / range_val) if range_val else 0.0 for tau in thresholds]
    min_sessions = min(session_counts)
    max_sessions = max(session_counts)
    if max_sessions == min_sessions:
        return math.exp(_median(thresholds))
    session_range = max_sessions - min_sessions
    norm_y = [
        (count - min_sessions) / session_range if session_range else 0.0
        for count in session_counts
    ]

    x0 = norm_x[0]
    y0 = norm_y[0]
    x1 = norm_x[-1]
    y1 = norm_y[-1]
    if math.isclose(x1, x0):
        slope = 0.0
    else:
        slope = (y1 - y0) / (x1 - x0)

    distances = [
        abs((ny - y0) - slope * (nx - x0)) / math.sqrt(1 + slope**2)
        for nx, ny in zip(norm_x, norm_y)
    ]
    best_index = max(range(len(distances)), key=lambda i: distances[i])
    return math.exp(thresholds[best_index])


def _count_sessions_for_threshold(
    timestamps_per_uid: dict[str, list[datetime]], threshold_seconds: float
) -> int:
    total_sessions = 0
    for times in timestamps_per_uid.values():
        if not times:
            continue
        count = 1
        for previous, current in zip(times, times[1:]):
            if (current - previous).total_seconds() > threshold_seconds:
                count += 1
        total_sessions += count
    return total_sessions


def _assign_sessions(
    rows: list[dict[str, str]], threshold_seconds: float
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row)

    for user_rows in grouped.values():
        user_rows.sort(key=lambda item: item["timestamp_utc"])

    sessionized: list[dict[str, str]] = []
    for uid, user_rows in grouped.items():
        current_session = 0
        previous_time: datetime | None = None
        for order, row in enumerate(user_rows, start=1):
            timestamp = datetime.fromisoformat(row["timestamp_utc"])
            if previous_time is None:
                delta_seconds = 0.0
                current_session += 1
            else:
                delta_seconds = max((timestamp - previous_time).total_seconds(), 0.0)
                if delta_seconds > threshold_seconds:
                    current_session += 1
            previous_time = timestamp

            session_row = dict(row)
            session_row["session_id"] = f"{uid}-{current_session:04d}"
            session_row["delta_t_seconds"] = f"{delta_seconds:.6f}"
            session_row["log_delta_t"] = f"{math.log(delta_seconds + EPSILON):.12f}"
            sessionized.append(session_row)

    sessionized.sort(key=lambda item: item["timestamp_utc"])
    return sessionized


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
            q25 = float(_quantile(history, 0.25))
            q75 = float(_quantile(history, 0.75))
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
