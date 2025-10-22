"""Data loading and feature preparation utilities for the LSTM models."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "uid",
    "session_id",
    "method",
    "path",
    "referer",
    "user_agent",
    "ip",
    "op_category",
)


@dataclass(frozen=True)
class FeatureStats:
    """Container for robust statistics over log-delta features."""

    log_median: float
    log_mad: float
    clip_min: float
    clip_max: float


@dataclass(frozen=True)
class FeaturePipelineResult:
    """Result of the feature preparation pipeline."""

    features: List[Dict[str, Any]]
    op_index_mapping: Dict[str, int]
    stats: FeatureStats


def _parse_timestamp(value: object) -> datetime:
    """Parse timestamp representations into timezone-aware UTC datetimes."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1e12:  # epoch milliseconds
            timestamp /= 1_000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return _parse_timestamp(int(stripped))
        if stripped.endswith("Z"):
            stripped = stripped[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(stripped)
        except ValueError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Invalid timestamp_utc value: {value!r}") from exc
        return _parse_timestamp(parsed)

    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _validate_records(records: Sequence[Mapping[str, object]]) -> None:
    """Ensure that all required columns are present in every record."""

    for index, record in enumerate(records):
        for column in REQUIRED_COLUMNS:
            if column not in record:
                raise ValueError(f"Missing required column '{column}' at row {index}.")


def _compute_log_deltas(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[float], List[float], float, float]:
    """Compute log-delta values and their robust statistics."""

    log_deltas: List[float] = []
    raw_deltas: List[float] = []
    last_seen: Dict[str, datetime] = {}

    for record in records:
        uid = str(record["uid"])
        session_id = str(record["session_id"])
        key = f"{uid}::{session_id}"
        timestamp = record["_parsed_ts"]
        previous = last_seen.get(key)
        if previous is None:
            delta_seconds = 0.0
        else:
            delta_seconds = (timestamp - previous).total_seconds()
            if delta_seconds < 0:
                raise ValueError(
                    "Timestamps must be non-decreasing within each session."
                    f" Found negative delta for group {key!r}."
                )
        last_seen[key] = timestamp
        raw_deltas.append(delta_seconds)
        log_deltas.append(math.log(delta_seconds + 1e-3))

    median = statistics.median(log_deltas)
    deviations = [abs(value - median) for value in log_deltas]
    mad = statistics.median(deviations)
    return log_deltas, raw_deltas, median, mad


def _build_op_mapping(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Build a deterministic mapping from op_category strings to indices."""

    categories = sorted({str(record["op_category"]) for record in records})
    return {category: index for index, category in enumerate(categories)}


def prepare_lstm_features(rows: Sequence[Mapping[str, Any]]) -> FeaturePipelineResult:
    """Prepare features for the LSTM model from contract CSV rows."""

    ordered_rows = [dict(row) for row in rows]
    _validate_records(ordered_rows)

    for row in ordered_rows:
        parsed_ts = _parse_timestamp(row["timestamp_utc"])
        row["_parsed_ts"] = parsed_ts
        row["_ts_iso"] = (
            parsed_ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        )

    ordered_rows.sort(
        key=lambda record: (
            str(record["uid"]),
            str(record["session_id"]),
            record["_parsed_ts"],
        )
    )

    log_deltas, raw_deltas, median, mad = _compute_log_deltas(ordered_rows)
    scale = 1.4826 * (mad if mad > 0 else 1.0)
    z_scores = [(value - median) / scale for value in log_deltas]
    clipped = [max(min(score, 5.0), -5.0) for score in z_scores]

    op_mapping = _build_op_mapping(ordered_rows)

    prepared: List[Dict[str, Any]] = []
    for index, row in enumerate(ordered_rows):
        feature_row: Dict[str, Any] = {
            column: row[column] for column in REQUIRED_COLUMNS
        }
        feature_row["timestamp_utc"] = row["_ts_iso"]
        feature_row["delta_t"] = raw_deltas[index]
        feature_row["log_delta"] = log_deltas[index]
        feature_row["z_score"] = z_scores[index]
        feature_row["z_clip"] = clipped[index]
        feature_row["op_index"] = op_mapping[str(row["op_category"])]
        prepared.append(feature_row)

    stats = FeatureStats(
        log_median=median,
        log_mad=mad,
        clip_min=-5.0,
        clip_max=5.0,
    )
    return FeaturePipelineResult(
        features=prepared, op_index_mapping=op_mapping, stats=stats
    )


def split_group_kfold(
    features: Sequence[Mapping[str, Any]],
    *,
    n_splits: int,
) -> Iterator[Tuple[List[int], List[int]]]:
    """Yield deterministic GroupKFold-style splits using uid/session pairs."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    group_to_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(features):
        if "uid" not in row or "session_id" not in row:
            raise ValueError("Each feature row must include 'uid' and 'session_id'.")
        group_key = f"{row['uid']}::{row['session_id']}"
        group_to_indices.setdefault(group_key, []).append(index)

    unique_groups = sorted(group_to_indices)
    if len(unique_groups) < n_splits:
        raise ValueError("Number of unique groups must be at least equal to n_splits.")

    fold_sizes = [len(unique_groups) // n_splits] * n_splits
    for i in range(len(unique_groups) % n_splits):
        fold_sizes[i] += 1

    folds: List[List[str]] = []
    cursor = 0
    for size in fold_sizes:
        folds.append(unique_groups[cursor : cursor + size])
        cursor += size

    for fold_index in range(n_splits):
        val_groups = set(folds[fold_index])
        train_indices: List[int] = []
        val_indices: List[int] = []
        for group, indices in group_to_indices.items():
            if group in val_groups:
                val_indices.extend(indices)
            else:
                train_indices.extend(indices)
        train_indices.sort()
        val_indices.sort()
        yield train_indices, val_indices
