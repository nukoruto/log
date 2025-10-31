"""Tests for contract CSV validation utilities."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import cast

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from log_generator.validate import (
    ContractValidationError,
    RowDict,
    _validate_row,
    validate_contract_csv,
)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def test_validate_contract_csv_accepts_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "normal.csv"
    header = [
        "timestamp_utc",
        "uid",
        "session_id",
        "method",
        "path",
        "referer",
        "user_agent",
        "ip",
        "op_category",
    ]
    _write_csv(
        path,
        header,
        [
            [
                "1970-01-01T00:00:00Z",
                "user-1",
                "sess-1",
                "GET",
                "/",
                "-",
                "ua",
                "127.0.0.1",
                "homepage",
            ]
        ],
    )

    validate_contract_csv(path)


def test_validate_contract_csv_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    header = [
        "timestamp_utc",
        "uid",
        "session_id",
        "method",
        "path",
        "referer",
        "user_agent",
        "ip",
    ]  # Missing op_category
    _write_csv(
        path,
        header,
        [
            [
                "1970-01-01T00:00:00Z",
                "user-1",
                "sess-1",
                "GET",
                "/",
                "-",
                "ua",
                "127.0.0.1",
            ]
        ],
    )

    with pytest.raises(ContractValidationError, match="missing columns"):
        validate_contract_csv(path)


def test_validate_contract_csv_rejects_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_extra.csv"
    header = [
        "timestamp_utc",
        "uid",
        "session_id",
        "method",
        "path",
        "referer",
        "user_agent",
        "ip",
        "op_category",
    ]
    _write_csv(
        path,
        header,
        [
            [
                "1970-01-01T00:00:00Z",
                "user-1",
                "sess-1",
                "GET",
                "/",
                "-",
                "ua",
                "127.0.0.1",
                "homepage",
                "unexpected",
            ]
        ],
    )

    with pytest.raises(ContractValidationError, match="unexpected"):
        validate_contract_csv(path)


def test_validate_contract_csv_rejects_non_utc_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "bad_ts.csv"
    header = [
        "timestamp_utc",
        "uid",
        "session_id",
        "method",
        "path",
        "referer",
        "user_agent",
        "ip",
        "op_category",
    ]
    _write_csv(
        path,
        header,
        [
            [
                "1970-01-01T00:00:00+09:00",
                "user-1",
                "sess-1",
                "GET",
                "/",
                "-",
                "ua",
                "127.0.0.1",
                "homepage",
            ]
        ],
    )

    with pytest.raises(ContractValidationError, match="UTC"):
        validate_contract_csv(path)


def test_validate_contract_csv_rejects_non_string_timestamp() -> None:
    row = cast(
        RowDict,
        {
            "timestamp_utc": 123,
            "uid": "user-1",
            "session_id": "sess-1",
            "method": "GET",
            "path": "/",
            "referer": "-",
            "user_agent": "ua",
            "ip": "127.0.0.1",
            "op_category": "homepage",
        },
    )

    with pytest.raises(
        ContractValidationError, match="timestamp_utc' must be a string .*got int"
    ):
        _validate_row(row, 2)
