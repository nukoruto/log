"""Core logic for converting scored CSV files to MATLAB MAT format."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class ExportResult:
    """Summary of the export process."""

    row_count: int
    input_sha256: str
    generated_at: datetime


@dataclass(frozen=True)
class SignalRow:
    """Single row of scored CSV data.

    The MATLAB bridge derives the control signals from the scored.csv
    contract columns as follows:
    - ref: combined anomaly score ``S``
    - y_lstm: temporal anomaly score ``s_time``
    - y_pid: PID baseline decision ``flag_dt`` (coerced to float)
    """

    timestamp_utc: datetime
    ref: float
    y_lstm: float
    y_pid: float


SCORED_HEADER = [
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
]


class ExportError(RuntimeError):
    """Raised when the export process encounters invalid input."""


def export_to_mat(
    csv_path: Path,
    mat_path: Path,
    *,
    meta_path: Path | None = None,
    meta_context: dict[str, object] | None = None,
) -> ExportResult:
    """Convert a scored CSV file to a MATLAB MAT v4 file.

    Args:
        csv_path: Path to the scored CSV file.
        mat_path: Destination path for the MAT file.

    Returns:
        ExportResult containing the processed row count and SHA256 hash.

    Raises:
        ExportError: If the CSV violates the contract or timestamps are invalid.
        FileNotFoundError: If the input CSV does not exist.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = _read_csv(csv_path)
    if not rows:
        raise ExportError("scored.csv must contain at least one record")

    (
        time_values,
        ref_values,
        lstm_values,
        pid_values,
    ) = _build_uniform_signals(rows)

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    write_mat_v4(
        mat_path,
        {
            "ref": ref_values,
            "y_lstm": lstm_values,
            "y_pid": pid_values,
            "t": time_values,
        },
    )

    result = ExportResult(
        row_count=len(rows),
        input_sha256=_sha256_digest(csv_path),
        generated_at=rows[-1].timestamp_utc,
    )

    if meta_path is not None:
        if meta_context is None:
            meta_context = {}
        _write_meta_json(meta_path, result, meta_context)

    return result


def _read_csv(csv_path: Path) -> list[SignalRow]:
    """Read the CSV file and validate required columns."""

    records: list[SignalRow] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ExportError("CSV header is missing")

        header = [field.strip() for field in reader.fieldnames]
        if header != SCORED_HEADER:
            raise ExportError(
                "scored.csv header must match "
                + ", ".join(SCORED_HEADER)
                + "; got "
                + ", ".join(reader.fieldnames)
            )

        for row in reader:
            timestamp = _parse_timestamp(row.get("timestamp_utc"))
            ref = _parse_float(row.get("S"), "S")
            y_lstm = _parse_float(row.get("s_time"), "s_time")
            y_pid = _parse_float(row.get("flag_dt"), "flag_dt")

            records.append(
                SignalRow(
                    timestamp_utc=timestamp,
                    ref=ref,
                    y_lstm=y_lstm,
                    y_pid=y_pid,
                )
            )

    return records


def _parse_timestamp(value: str | None) -> datetime:
    """Parse an ISO8601 timestamp ensuring UTC timezone."""

    if value is None:
        raise ExportError("timestamp_utc column contains null entries")

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ExportError(f"Invalid ISO8601 timestamp: {value}") from exc

    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) != UTC.utcoffset(None):
        raise ExportError("timestamp_utc must be timezone-aware in UTC")

    return parsed.astimezone(UTC)


def _parse_float(value: str | None, column: str) -> float:
    """Parse a floating-point value ensuring finiteness."""

    if value is None:
        raise ExportError(f"{column} column contains null entries")

    text = value.strip()
    if not text:
        raise ExportError(f"{column} column contains empty values")

    try:
        numeric = float(text)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{column} must be a finite float; got {value!r}") from exc

    if not math.isfinite(numeric):
        raise ExportError(f"{column} must be a finite float; got {value!r}")

    return numeric


def _build_uniform_signals(
    rows: list[SignalRow],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Generate a median-Δt grid and resample signals accordingly."""

    times, ref_values, lstm_values, pid_values = _extract_series(rows)

    if len(times) == 1:
        return times, ref_values, lstm_values, pid_values

    grid = _compute_time_axis(times)
    return (
        grid,
        _resample_to_grid(times, ref_values, grid),
        _resample_to_grid(times, lstm_values, grid),
        _resample_to_grid(times, pid_values, grid),
    )


def _extract_series(
    rows: list[SignalRow],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Extract monotonic timestamps and associated signal values."""

    base = rows[0].timestamp_utc
    times = [0.0]
    ref_values = [rows[0].ref]
    lstm_values = [rows[0].y_lstm]
    pid_values = [rows[0].y_pid]
    last_timestamp = base

    for row in rows[1:]:
        timestamp = row.timestamp_utc
        if timestamp <= last_timestamp:
            raise ExportError(
                "t must be strictly increasing; check timestamp_utc ordering"
            )

        offset = (timestamp - base).total_seconds()
        times.append(offset)
        ref_values.append(row.ref)
        lstm_values.append(row.y_lstm)
        pid_values.append(row.y_pid)
        last_timestamp = timestamp

    return times, ref_values, lstm_values, pid_values


def _compute_time_axis(times: list[float]) -> list[float]:
    """Compute an evenly spaced time axis using the median Δt."""

    deltas = [current - previous for previous, current in zip(times[:-1], times[1:])]
    step = median(deltas)
    if step <= 0:
        raise ExportError("Median Δt must be positive")

    return [index * step for index in range(len(times))]


def _resample_to_grid(
    times: list[float], values: list[float], grid: list[float]
) -> list[float]:
    """Linearly interpolate values onto the provided grid."""

    if len(times) != len(values):  # pragma: no cover - defensive
        raise ExportError("Time and value arrays must be the same length")

    if len(times) == 1:
        return values.copy()

    result: list[float] = []
    last_index = len(times) - 1
    index = 0

    for point in grid:
        while index < last_index and point > times[index + 1]:
            index += 1

        if point <= times[0]:
            result.append(values[0])
            continue

        if point >= times[last_index]:
            result.append(values[last_index])
            continue

        next_index = index + 1
        start_time = times[index]
        end_time = times[next_index]
        start_value = values[index]
        end_value = values[next_index]

        if end_time == start_time:  # pragma: no cover - defensive
            result.append(end_value)
            continue

        ratio = (point - start_time) / (end_time - start_time)
        interpolated = start_value + ratio * (end_value - start_value)
        result.append(interpolated)

    return result


def write_mat_v4(path: Path, variables: dict[str, Iterable[float]]) -> None:
    """Write variables to a MATLAB v4 MAT file."""

    with path.open("wb") as handle:
        for name in ("ref", "y_lstm", "y_pid", "t"):
            values = [float(v) for v in variables[name]]
            _write_matrix(handle, name, values)


def _write_matrix(handle, name: str, values: list[float]) -> None:
    """Write a single column vector matrix to the MAT file handle."""

    import struct

    if not name.isidentifier():  # pragma: no cover - defensive
        raise ExportError(f"Invalid MATLAB variable name: {name}")

    name_bytes = name.encode("ascii") + b"\x00"
    namelen = len(name_bytes)
    mrows = len(values)
    ncols = 1
    header = struct.pack("<5i", 0, mrows, ncols, 0, namelen)
    handle.write(header)
    handle.write(name_bytes)

    for value in values:
        handle.write(struct.pack("<d", float(value)))


def _sha256_digest(path: Path) -> str:
    """Compute the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_meta_json(
    path: Path, result: ExportResult, context: dict[str, object]
) -> None:
    """Persist export metadata to a JSON file."""

    seed = context.get("seed")
    algo_version = context.get("algo_version")
    if seed is None or algo_version is None:
        raise ExportError("meta_context must include seed and algo_version")

    output_raw = context.get("output")
    if isinstance(output_raw, (str, Path)) and output_raw:
        output_name = Path(output_raw).name
    else:
        output_name = ""

    payload = {
        "seed": seed,
        "algo_version": algo_version,
        "input_sha256": result.input_sha256,
        "row_count": result.row_count,
        "generated_at": result.generated_at.isoformat().replace("+00:00", "Z"),
        "output_mat": output_name,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
