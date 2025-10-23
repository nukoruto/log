"""Data loading and preprocessing utilities for contract CSV inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Sequence,
    Tuple,
    Union,
)

REQUIRED_CONTRACT_COLUMNS: Tuple[str, ...] = (
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
MAD_SCALE: float = 1.4826
EPSILON: float = 1e-8
GroupKey = Union[Tuple[str, str], str]


@dataclass
class ContractRecord:
    """Typed representation of a contract CSV row."""

    timestamp_utc: datetime
    uid: str
    session_id: str
    method: str
    path: str
    referer: str
    user_agent: str
    ip: str
    op_category: str
    delta_seconds: float = 0.0
    z_score: float = 0.0
    z_clipped: float = 0.0


@dataclass(frozen=True)
class SequenceExample:
    """Container for a single user session sequence."""

    uid: str
    session_id: str
    timestamps: Tuple[datetime, ...]
    op_indices: Tuple[int, ...]
    delta_seconds: Tuple[float, ...]
    z_clipped: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamps", tuple(self.timestamps))
        object.__setattr__(
            self, "op_indices", tuple(int(value) for value in self.op_indices)
        )
        object.__setattr__(
            self, "delta_seconds", tuple(float(value) for value in self.delta_seconds)
        )
        object.__setattr__(
            self, "z_clipped", tuple(float(value) for value in self.z_clipped)
        )


class OpCategoryEncoder:
    """Deterministic encoder for operation categories."""

    def __init__(self) -> None:
        self._category_to_index: Dict[str, int] = {}
        self._index_to_category: List[str] = []

    @property
    def vocab_size(self) -> int:
        """Return the number of known categories."""

        return len(self._index_to_category)

    def fit(self, categories: Iterable[str]) -> "OpCategoryEncoder":
        """Fit the encoder using all observed categories."""

        unique = sorted(dict.fromkeys(categories))
        self._index_to_category = list(unique)
        self._category_to_index = {
            category: index for index, category in enumerate(self._index_to_category)
        }
        return self

    def transform(self, categories: Sequence[str]) -> Tuple[int, ...]:
        """Convert categories into indices."""

        if not self._category_to_index:
            raise RuntimeError("Encoder has not been fitted.")

        indices: List[int] = []
        for category in categories:
            try:
                indices.append(self._category_to_index[category])
            except KeyError as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unknown category encountered: {category}") from exc
        return tuple(indices)

    def inverse_transform(self, indices: Sequence[int]) -> Tuple[str, ...]:
        """Map indices back to their categories."""

        return tuple(self._index_to_category[index] for index in indices)


def load_contract_dataframe(path: Path | str) -> List[Dict[str, str]]:
    """Load and validate the contract CSV returning a list of records."""

    path = Path(path)
    if not path.exists():  # pragma: no cover - defensive check
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV is missing a header row.")
        expected = list(REQUIRED_CONTRACT_COLUMNS)
        missing = [column for column in expected if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        unexpected = [column for column in fieldnames if column not in expected]
        if unexpected or fieldnames != expected:
            problems: List[str] = []
            if unexpected:
                problems.append(f"unexpected columns: {', '.join(unexpected)}")
            if fieldnames != expected:
                problems.append(
                    "header order must exactly match contract specification"
                )
                problems.append(
                    f"expected header order: {', '.join(expected)}; got: {', '.join(fieldnames)}"
                )
            raise ValueError(
                "Invalid contract CSV header: " + "; ".join(dict.fromkeys(problems))
            )
        rows = [dict(row) for row in reader]
    return rows


def load_contract_sequences(
    path: Path | str,
    *,
    clip_value: float = 5.0,
    encoder: OpCategoryEncoder | None = None,
) -> Tuple[List[SequenceExample], OpCategoryEncoder, Dict[str, float]]:
    """Load contract CSV and build session sequences."""

    records = load_contract_dataframe(path)
    return build_sequence_examples(records, clip_value=clip_value, encoder=encoder)


def build_sequence_examples(
    records: Sequence[Mapping[str, object]],
    *,
    clip_value: float = 5.0,
    encoder: OpCategoryEncoder | None = None,
) -> Tuple[List[SequenceExample], OpCategoryEncoder, Dict[str, float]]:
    """Create session sequences from validated contract records."""

    prepared = _prepare_records(records)
    encoder = encoder or OpCategoryEncoder()
    encoder.fit(record.op_category for record in prepared)

    sorted_records = sorted(
        prepared,
        key=lambda record: (
            record.uid,
            record.session_id,
            record.timestamp_utc,
        ),
    )

    delta_values = _compute_delta_seconds(sorted_records)
    z_scores, z_clipped, stats = _compute_robust_z(delta_values, clip_value=clip_value)

    for record, z_value, clipped_value in zip(sorted_records, z_scores, z_clipped):
        record.z_score = z_value
        record.z_clipped = clipped_value

    grouped: Dict[Tuple[str, str], List[ContractRecord]] = {}
    for record in sorted_records:
        key = (record.uid, record.session_id)
        grouped.setdefault(key, []).append(record)

    sequences: List[SequenceExample] = []
    for (uid, session_id), session_records in grouped.items():
        timestamps = tuple(record.timestamp_utc for record in session_records)
        categories = [record.op_category for record in session_records]
        op_indices = encoder.transform(tuple(categories))
        delta = tuple(record.delta_seconds for record in session_records)
        clipped_values = tuple(record.z_clipped for record in session_records)
        sequences.append(
            SequenceExample(
                uid=uid,
                session_id=session_id,
                timestamps=timestamps,
                op_indices=op_indices,
                delta_seconds=delta,
                z_clipped=clipped_values,
            ),
        )

    stats.update({"clip_value": float(clip_value)})
    return sequences, encoder, stats


def group_kfold_split(
    sequences: Sequence[SequenceExample],
    *,
    n_splits: int,
    group_level: str = "session",
) -> Iterator[Tuple[List[int], List[int]]]:
    """Yield deterministic GroupKFold-style splits based on the requested granularity."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not sequences:
        raise ValueError("sequences must not be empty")

    if group_level == "session":
        group_keys: List[GroupKey] = [(seq.uid, seq.session_id) for seq in sequences]
    elif group_level == "user":
        group_keys = [seq.uid for seq in sequences]
    else:
        raise ValueError(f"Unsupported group_level: {group_level}")

    group_to_indices: Dict[GroupKey, List[int]] = {}
    ordered_groups: List[GroupKey] = []
    for index, key in enumerate(group_keys):
        if key not in group_to_indices:
            group_to_indices[key] = []
            ordered_groups.append(key)
        group_to_indices[key].append(index)

    if len(ordered_groups) < n_splits:
        raise ValueError("Number of unique groups must be at least n_splits")

    for fold in range(n_splits):
        test_groups = [
            ordered_groups[position]
            for position in range(fold, len(ordered_groups), n_splits)
        ]
        test_group_set = set(test_groups)
        test_indices = [
            index for group in test_groups for index in group_to_indices[group]
        ]
        train_indices = [
            index
            for group in ordered_groups
            if group not in test_group_set
            for index in group_to_indices[group]
        ]
        yield train_indices, test_indices


def _prepare_records(
    records: Sequence[Mapping[str, object]],
) -> List[ContractRecord]:
    prepared: List[ContractRecord] = []
    for record in records:
        missing = [
            column for column in REQUIRED_CONTRACT_COLUMNS if column not in record
        ]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        timestamp = _parse_timestamp(record["timestamp_utc"])
        prepared.append(
            ContractRecord(
                timestamp_utc=timestamp,
                uid=str(record["uid"]),
                session_id=str(record["session_id"]),
                method=str(record["method"]),
                path=str(record["path"]),
                referer=str(record["referer"]),
                user_agent=str(record["user_agent"]),
                ip=str(record["ip"]),
                op_category=str(record["op_category"]),
            ),
        )
    return prepared


def _parse_timestamp(value: object) -> datetime:
    def _raise_invalid(reason: str) -> NoReturn:
        raise ValueError(
            "Timestamp must be timezone-aware UTC (+00:00); "
            f"{reason}. Please re-export UTC-normalized contract data."
        )

    if isinstance(value, datetime):
        offset = value.utcoffset()
        if offset is None:
            _raise_invalid("received naive datetime without timezone information")
        if offset.total_seconds() != 0:
            hours = offset.total_seconds() / 3600
            _raise_invalid(f"received non-UTC offset {hours:+g}h")
        return value.astimezone(timezone.utc)

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    offset = parsed.utcoffset()
    if offset is None:
        _raise_invalid("timestamp lacks timezone offset")
    if offset.total_seconds() != 0:
        hours = offset.total_seconds() / 3600
        _raise_invalid(f"received non-UTC offset {hours:+g}h")
    return parsed.astimezone(timezone.utc)


def _compute_delta_seconds(
    records: Sequence[ContractRecord],
) -> List[float]:
    last_timestamp: Dict[Tuple[str, str], datetime] = {}
    delta_seconds: List[float] = []
    for record in records:
        key = (record.uid, record.session_id)
        timestamp = record.timestamp_utc
        previous = last_timestamp.get(key)
        if previous is None:
            delta_seconds.append(0.0)
            record.delta_seconds = 0.0
        else:
            delta = max(0.0, (timestamp - previous).total_seconds())
            delta_seconds.append(delta)
            record.delta_seconds = delta
        last_timestamp[key] = timestamp
    return delta_seconds


def _compute_robust_z(
    delta_seconds: Sequence[float], *, clip_value: float
) -> Tuple[List[float], List[float], Dict[str, float]]:
    values = list(float(value) for value in delta_seconds)
    if not values:
        raise ValueError("delta_seconds must not be empty")

    median_value = _median(values)
    deviations = [abs(value - median_value) for value in values]
    mad_value = _median(deviations)
    scale = MAD_SCALE * mad_value
    if scale <= 0:
        scale = 1.0

    z_scores = [(value - median_value) / max(scale, EPSILON) for value in values]
    clipped = [min(clip_value, max(-clip_value, score)) for score in z_scores]
    stats = {
        "median_delta_seconds": float(median_value),
        "mad_delta_seconds": float(mad_value),
        "robust_scale": float(scale),
    }
    return z_scores, clipped, stats


def _median(values: Sequence[float]) -> float:
    sorted_values = sorted(values)
    length = len(sorted_values)
    middle = length // 2
    if length % 2 == 1:
        return float(sorted_values[middle])
    return float((sorted_values[middle - 1] + sorted_values[middle]) / 2)


__all__ = [
    "SequenceExample",
    "OpCategoryEncoder",
    "build_sequence_examples",
    "group_kfold_split",
    "load_contract_dataframe",
    "load_contract_sequences",
]
