"""validate サブコマンドの詳細テスト。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_contract import cli


def _write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_logs(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_mapping(path: Path, mapping: dict[str, str]) -> None:
    lines = [f"{target}: {source}" for target, source in mapping.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_accepts_epoch_ms_and_timezone_and_emits_sorted_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta.json"

    _write_csv(
        raw_path,
        fieldnames=[
            "timestamp",
            "user",
            "session",
            "method",
            "path",
            "referer",
            "user_agent",
            "ip",
            "category",
        ],
        rows=[
            {
                "timestamp": "2024-01-02T09:15:00+09:00",
                "user": "user-iso",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "ip": "198.51.100.2",
                "category": "read",
            },
            {
                "timestamp": "1704154200000",  # 2024-01-02T00:10:00Z in ms
                "user": "user-epoch",
                "session": "s-2",
                "method": "POST",
                "path": "/submit",
                "referer": "https://ref",
                "user_agent": "ua",
                "ip": "198.51.100.1",
                "category": "write",
            },
        ],
    )

    _write_mapping(
        map_path,
        {
            "timestamp_utc": "timestamp",
            "uid": "user",
            "session_id": "session",
            "method": "method",
            "path": "path",
            "referer": "referer",
            "user_agent": "user_agent",
            "ip": "ip",
            "op_category": "category",
        },
    )

    exit_code = cli.main(
        [
            "--seed",
            "11",
            "validate",
            str(raw_path),
            "--map",
            str(map_path),
            "--out",
            str(contract_path),
            "--meta",
            str(meta_path),
        ]
    )
    assert exit_code == 0
    logs = _load_logs(capsys.readouterr().out)
    assert logs[0]["event"] == "start"
    assert logs[-1]["event"] == "complete"

    expected_bytes = (
        b"timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category\r\n"
        b"2024-01-02T00:10:00+00:00,user-epoch,s-2,POST,/submit,https://ref,ua,198.51.100.1,write\r\n"
        b"2024-01-02T00:15:00+00:00,user-iso,s-1,GET,/index,-,ua,198.51.100.2,read\r\n"
    )
    assert contract_path.read_bytes() == expected_bytes

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["row_count"] == 2
    assert meta["mapping"]["timestamp_utc"] == "timestamp"


def test_validate_without_meta_argument_uses_default_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"

    _write_csv(
        raw_path,
        fieldnames=[
            "timestamp",
            "user",
            "session",
            "method",
            "path",
            "referer",
            "user_agent",
            "ip",
            "category",
        ],
        rows=[
            {
                "timestamp": "2024-01-01T12:00:00+00:00",
                "user": "user-iso",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "ip": "198.51.100.2",
                "category": "read",
            }
        ],
    )

    _write_mapping(
        map_path,
        {
            "timestamp_utc": "timestamp",
            "uid": "user",
            "session_id": "session",
            "method": "method",
            "path": "path",
            "referer": "referer",
            "user_agent": "user_agent",
            "ip": "ip",
            "op_category": "category",
        },
    )

    exit_code = cli.main(
        [
            "--seed",
            "19",
            "validate",
            str(raw_path),
            "--map",
            str(map_path),
            "--out",
            str(contract_path),
        ]
    )
    assert exit_code == 0

    logs = _load_logs(capsys.readouterr().out)
    assert logs[0]["event"] == "start"
    assert logs[-1]["event"] == "complete"
    default_meta_path = contract_path.with_name(f"{contract_path.stem}.meta.json")
    assert default_meta_path.exists()

    meta = json.loads(default_meta_path.read_text(encoding="utf-8"))
    assert meta["row_count"] == 1
    assert meta["mapping"]["timestamp_utc"] == "timestamp"


def test_validate_allows_missing_optional_ip_mapping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta.json"

    _write_csv(
        raw_path,
        fieldnames=[
            "time",
            "user",
            "session",
            "method",
            "path",
            "referer",
            "user_agent",
            "category",
        ],
        rows=[
            {
                "time": "2024-01-01T00:00:00Z",
                "user": "user-1",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "category": "view",
            }
        ],
    )

    _write_mapping(
        map_path,
        {
            "timestamp_utc": "time",
            "uid": "user",
            "session_id": "session",
            "method": "method",
            "path": "path",
            "referer": "referer",
            "user_agent": "user_agent",
            "op_category": "category",
        },
    )

    exit_code = cli.main(
        [
            "--seed",
            "13",
            "validate",
            str(raw_path),
            "--map",
            str(map_path),
            "--out",
            str(contract_path),
            "--meta",
            str(meta_path),
        ]
    )
    assert exit_code == 0
    logs = _load_logs(capsys.readouterr().out)
    assert logs[-1]["event"] == "complete"

    with contract_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    assert row["ip"] == ""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "ip" not in meta["mapping"]


def test_validate_missing_required_value_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta.json"

    _write_csv(
        raw_path,
        fieldnames=[
            "time",
            "user",
            "session",
            "method",
            "path",
            "referer",
            "user_agent",
            "ip",
            "category",
        ],
        rows=[
            {
                "time": "2024-01-01T00:00:00Z",
                "user": "user-1",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "",  # missing required value
                "user_agent": "ua",
                "ip": "198.51.100.1",
                "category": "view",
            }
        ],
    )

    _write_mapping(
        map_path,
        {
            "timestamp_utc": "time",
            "uid": "user",
            "session_id": "session",
            "method": "method",
            "path": "path",
            "referer": "referer",
            "user_agent": "user_agent",
            "ip": "ip",
            "op_category": "category",
        },
    )

    exit_code = cli.main(
        [
            "--seed",
            "17",
            "validate",
            str(raw_path),
            "--map",
            str(map_path),
            "--out",
            str(contract_path),
            "--meta",
            str(meta_path),
        ]
    )
    assert exit_code == 1
    logs = _load_logs(capsys.readouterr().out)
    assert logs[-1]["event"] == "error"
    assert logs[-1]["code"] == "MISSING_REQUIRED_VALUE"
    assert "Row 1" in str(logs[-1]["message"])
    assert not contract_path.exists()
    assert not meta_path.exists()


def test_validate_rejects_naive_timestamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta.json"

    _write_csv(
        raw_path,
        fieldnames=[
            "timestamp",
            "user",
            "session",
            "method",
            "path",
            "referer",
            "user_agent",
            "ip",
            "category",
        ],
        rows=[
            {
                "timestamp": "2024-01-01T00:00:00",  # missing timezone info
                "user": "user-1",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "ip": "198.51.100.1",
                "category": "view",
            }
        ],
    )

    _write_mapping(
        map_path,
        {
            "timestamp_utc": "timestamp",
            "uid": "user",
            "session_id": "session",
            "method": "method",
            "path": "path",
            "referer": "referer",
            "user_agent": "user_agent",
            "ip": "ip",
            "op_category": "category",
        },
    )

    exit_code = cli.main(
        [
            "--seed",
            "23",
            "validate",
            str(raw_path),
            "--map",
            str(map_path),
            "--out",
            str(contract_path),
            "--meta",
            str(meta_path),
        ]
    )
    assert exit_code == 1

    logs = _load_logs(capsys.readouterr().out)
    assert logs[-1]["event"] == "error"
    assert logs[-1]["code"] == "INVALID_TIMESTAMP"
    assert "2024-01-01T00:00:00" in str(logs[-1]["message"])
    assert "hint" in logs[-1]
    assert "ISO8601" in str(logs[-1]["hint"])
    assert not contract_path.exists()
    assert not meta_path.exists()
