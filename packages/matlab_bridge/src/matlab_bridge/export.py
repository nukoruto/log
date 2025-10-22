"""Utilities for exporting scored data to MATLAB format."""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class ExportSummary:
    """Summary of an export run."""

    row_count: int
    t_start: float
    t_end: float


class ExportError(ValueError):
    """Raised when the input CSV violates export requirements."""


REQUIRED_COLUMNS: tuple[str, ...] = ("t", "ref", "y_lstm", "y_pid")

MI_INT8 = 1
MI_UINT32 = 6
MI_INT32 = 5
MI_DOUBLE = 9
MI_MATRIX = 14
MX_DOUBLE_CLASS = 6


def _read_csv(path: Path) -> dict[str, list[float]]:
    """Read the required columns from the scored CSV.

    Args:
        path: Path to the scored CSV.

    Returns:
        Mapping of column name to list of float values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ExportError: If required columns are missing or values are invalid.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ExportError("Input CSV is missing a header row.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ExportError(f"Missing required columns: {', '.join(missing)}")

        values: dict[str, list[float]] = {col: [] for col in REQUIRED_COLUMNS}
        for row_index, row in enumerate(reader, start=1):
            for column in REQUIRED_COLUMNS:
                raw_value = row.get(column, "")
                if raw_value is None or raw_value == "":
                    raise ExportError(
                        f"Empty value in column '{column}' at row {row_index}."
                    )
                try:
                    values[column].append(float(raw_value))
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ExportError(
                        f"Non-numeric value in column '{column}' at row {row_index}: {raw_value!r}"
                    ) from exc

    if not values["t"]:
        raise ExportError("Input CSV contains no data rows.")

    return values


def _ensure_strictly_increasing(sequence: Iterable[float]) -> None:
    """Validate that the provided sequence is strictly increasing."""

    iterator = iter(sequence)
    try:
        previous = next(iterator)
    except StopIteration:  # pragma: no cover - handled earlier
        return

    for current in iterator:
        if current <= previous:
            raise ExportError("Time column 't' must be strictly increasing.")
        previous = current


def export_to_mat(input_path: Path, output_path: Path) -> ExportSummary:
    """Export scored CSV to a MATLAB MAT file.

    Args:
        input_path: Location of the scored CSV file.
        output_path: Destination MAT file path.

    Returns:
        Summary of the exported data.

    Raises:
        FileNotFoundError: If the input file is absent.
        ExportError: If the CSV structure or data is invalid.
    """

    values: dict[str, list[float]] = _read_csv(input_path)
    _ensure_strictly_increasing(values["t"])

    row_count = len(values["t"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mat_file(output_path, values)

    return ExportSummary(
        row_count=row_count,
        t_start=values["t"][0],
        t_end=values["t"][-1],
    )


def _write_mat_file(output_path: Path, values: dict[str, list[float]]) -> None:
    """Write the collected values to a MATLAB Level-5 MAT file."""

    header_text = "MATLAB 5.0 MAT-file, Created by matlab_bridge"
    header_bytes = header_text.encode("utf-8")[:116]
    header = header_bytes.ljust(116, b" ")
    header += b"\x00" * 8  # Subsystem data offset
    header += struct.pack("<H2s", 0x0100, b"IM")

    with output_path.open("wb") as handle:
        handle.write(header)
        for name in REQUIRED_COLUMNS:
            dataset = _create_matrix_element(name, values[name])
            handle.write(dataset)


def _create_matrix_element(name: str, data: list[float]) -> bytes:
    """Create a MAT-file element representing a column vector of doubles."""

    rows = len(data)
    cols = 1
    real_data = b"".join(struct.pack("<d", float(value)) for value in data)

    content = b"".join(
        [
            _data_element(MI_UINT32, struct.pack("<II", MX_DOUBLE_CLASS, 0)),
            _data_element(MI_INT32, struct.pack("<ii", rows, cols)),
            _data_element(MI_INT8, name.encode("utf-8")),
            _data_element(MI_DOUBLE, real_data),
        ]
    )

    padding = _padding_needed(len(content))
    return struct.pack("<II", MI_MATRIX, len(content)) + content + (b"\x00" * padding)


def _data_element(mi_type: int, payload: bytes) -> bytes:
    """Build a tagged MAT-file data element with padding."""

    padding = _padding_needed(len(payload))
    return struct.pack("<II", mi_type, len(payload)) + payload + (b"\x00" * padding)


def _padding_needed(length: int) -> int:
    """Return the padding (in bytes) required to reach the next 8-byte boundary."""

    remainder = length % 8
    return 0 if remainder == 0 else 8 - remainder
