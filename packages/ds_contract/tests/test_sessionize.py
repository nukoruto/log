"""Unit tests for sessionize module."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_contract import cli
from ds_contract.contract import CONTRACT_COLUMNS
from ds_contract.sessionize import (
    SESSION_COLUMNS,
    SessionizationResult,
    sessionize_contract,
)


@pytest.fixture()
def base_rows() -> list[dict[str, str]]:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, str]] = []
    for offset in range(6):
        rows.append(_row(base_time + timedelta(seconds=30 * offset), "user-a"))
    long_start = base_time + timedelta(hours=2)
    for offset in range(6):
        rows.append(_row(long_start + timedelta(minutes=10 * offset), "user-a"))
    return rows


def _row(timestamp: datetime, uid: str) -> dict[str, str]:
    return {
        "timestamp_utc": timestamp.isoformat(),
        "uid": uid,
        "session_id": "raw",
        "method": "GET",
        "path": "/index",
        "referer": "-",
        "user_agent": "ua",
        "ip": "192.0.2.1",
        "op_category": "view",
    }


def test_sessionize_contract_returns_otsu_for_bimodal(
    base_rows: list[dict[str, str]],
) -> None:
    result = sessionize_contract(base_rows)

    assert isinstance(result, SessionizationResult)
    assert result.method == "otsu"
    assert result.threshold_seconds > 30
    assert result.within_class_variance_ratio < 0.9
    assert len(result.rows) == len(base_rows)
    session_ids = [row["session_id"] for row in result.rows]
    assert session_ids[0] == session_ids[5]
    assert session_ids[6] != session_ids[5]


def test_sessionize_contract_elbow_fallback() -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        _row(base_time + timedelta(minutes=10 * index), "user-b") for index in range(6)
    ]

    result = sessionize_contract(rows)

    assert result.method == "elbow"
    assert pytest.approx(result.threshold_seconds, rel=1e-4) == 600.0
    assert 0.0 <= result.within_class_variance_ratio <= 1.0


def test_sessionize_meta_contains_bins(base_rows: list[dict[str, str]]) -> None:
    result = sessionize_contract(base_rows)

    assert set(result.rows[0].keys()) >= set(SESSION_COLUMNS)
    histogram = result.histogram
    assert histogram["bin_count"] >= 32
    assert len(histogram["counts"]) == histogram["bin_count"]
    bins = result.bins
    assert bins["count"] == histogram["bin_count"]
    assert "fd_width" in bins
    assert "min_log" in bins and "max_log" in bins


def test_sessionize_cli_rejects_descending_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta_session.json"

    rows = [
        {
            "timestamp_utc": "2024-01-01T00:02:00+00:00",
            "uid": "user-1",
            "session_id": "raw",
            "method": "GET",
            "path": "/index",
            "referer": "-",
            "user_agent": "ua",
            "ip": "192.0.2.1",
            "op_category": "view",
        },
        {
            "timestamp_utc": "2024-01-01T00:01:00+00:00",
            "uid": "user-2",
            "session_id": "raw",
            "method": "POST",
            "path": "/submit",
            "referer": "-",
            "user_agent": "ua",
            "ip": "192.0.2.2",
            "op_category": "write",
        },
    ]

    with contract_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONTRACT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    exit_code = cli.main(
        [
            "--seed",
            "7",
            "sessionize",
            str(contract_path),
            "--out",
            str(tmp_path / "sessioned.csv"),
            "--meta",
            str(meta_path),
        ]
    )

    assert exit_code == 1
    logs = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert logs[-1]["event"] == "error"
    assert logs[-1]["code"] == "REVERSED_ORDER"
    assert "timestamp_utc" in str(logs[-1]["message"]).lower()
