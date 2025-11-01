"""Validation utilities ensuring generated CSV outputs honour the contract."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .generate import CONTRACT_COLUMNS


RowDict = dict[str | None, str | list[str] | None]


class ContractValidationError(ValueError):
    """Raised when a generated CSV file violates the contract."""


def validate_contract_csv(path: Path) -> None:
    """Validate that the CSV at ``path`` matches the contract requirements."""

    if not path.exists():
        raise ContractValidationError(f"contract csv not found: {path}")

    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        header = reader.fieldnames
        expected = list(CONTRACT_COLUMNS)
        if header is None:
            raise ContractValidationError("contract csv is empty or missing header row")
        if header != expected:
            missing = _difference(expected, header)
            extra = _difference(header, expected)
            parts: list[str] = []
            if missing:
                parts.append(f"missing columns: {missing}")
            if extra:
                parts.append(f"unexpected columns: {extra}")
            details = (
                "; ".join(parts)
                if parts
                else f"expected header {expected}, found {header}"
            )
            raise ContractValidationError(details)

        for line_number, row in enumerate(reader, start=2):
            _validate_row(row, line_number)


def _difference(left: Iterable[str], right: Iterable[str]) -> list[str]:
    return [item for item in left if item not in right]


def _validate_row(row: RowDict, line_number: int) -> None:
    extra_values_raw = row.get(None)
    if extra_values_raw:
        extras = (
            list(extra_values_raw)
            if isinstance(extra_values_raw, list)
            else [str(extra_values_raw)]
        )
        raise ContractValidationError(
            (
                f"row {line_number}: unexpected extra column values {extras} "
                f"(ensure exactly {len(CONTRACT_COLUMNS)} columns)"
            )
        )

    unexpected_keys = [
        key for key in row.keys() if key is not None and key not in CONTRACT_COLUMNS
    ]
    if unexpected_keys:
        raise ContractValidationError(
            (
                f"row {line_number}: unexpected columns {unexpected_keys} "
                f"(expected {list(CONTRACT_COLUMNS)})"
            )
        )

    for column in CONTRACT_COLUMNS:
        value = row.get(column)
        if isinstance(value, list):
            raise ContractValidationError(
                (
                    f"row {line_number}: column '{column}' contains multiple values "
                    f"{value}"
                )
            )
        if value is None:
            raise ContractValidationError(
                f"row {line_number}: column '{column}' is missing (check CSV delimiter)"
            )
        if value == "":
            raise ContractValidationError(
                f"row {line_number}: column '{column}' is empty"
            )

    timestamp_value = row.get("timestamp_utc")
    if not isinstance(timestamp_value, str):
        actual_type = type(timestamp_value).__name__
        raise ContractValidationError(
            (
                "row "
                f"{line_number}: column 'timestamp_utc' must be a string "
                f"(got {actual_type})"
            )
        )

    _validate_timestamp(timestamp_value, line_number)


def _validate_timestamp(raw: str, line_number: int) -> None:
    candidate = raw
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ContractValidationError(
            f"row {line_number}: timestamp_utc '{raw}' is not valid ISO8601"
        ) from exc

    if parsed.tzinfo is None:
        raise ContractValidationError(
            f"row {line_number}: timestamp_utc '{raw}' is missing UTC offset"
        )

    if parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ContractValidationError(
            f"row {line_number}: timestamp_utc '{raw}' must be in UTC"
        )


__all__ = ["ContractValidationError", "validate_contract_csv"]
