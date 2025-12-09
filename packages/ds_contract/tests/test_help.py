"""ds_contract CLI end-to-end tests for validate/sessionize/deltify."""

from __future__ import annotations

import csv
import json
import hashlib
import hmac
from pathlib import Path
from typing import Any, Iterable, cast

import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_contract import cli


CONTRACT_COLUMNS = [
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


def _expected_uid(ip: str, user_agent: str, seed: int) -> str:
    key = str(seed).encode("utf-8")
    message = f"{ip}|{user_agent}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.strip().splitlines() if line.strip()]


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--help"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage: ds-contract" in captured.out


def test_validate_produces_contract_csv_and_meta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta_contract.json"

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
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.1",
                "category": "view",
            },
            {
                "time": "2024-01-01T00:05:00Z",
                "user": "user-2",
                "session": "s-2",
                "method": "POST",
                "path": "/submit",
                "referer": "https://example",
                "user_agent": "ua",
                "ip": "192.0.2.2",
                "category": "submit",
            },
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--seed",
            "42",
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
    captured = capsys.readouterr()
    logs = _load_json_lines(captured.out)
    assert logs[0]["event"] == "start"
    assert logs[-1]["event"] == "complete"

    with contract_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header == CONTRACT_COLUMNS
        first_row = next(reader)
    assert first_row[0] == "2024-01-01T00:00:00+00:00"
    assert first_row[1] == _expected_uid("192.0.2.1", "ua", seed=42)
    meta_data: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_data["row_count"] == 2
    assert meta_data["input_sha256"].isalnum() and len(meta_data["input_sha256"]) == 64
    assert sorted(meta_data["mapping"].keys()) == sorted(CONTRACT_COLUMNS)


def test_validate_missing_required_column_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta_contract.json"

    _write_csv(
        raw_path,
        fieldnames=[
            "time",
            "user",
            "session",
            "method",
            "path",
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
                "user_agent": "ua",
                "ip": "192.0.2.1",
                "category": "view",
            }
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--seed",
            "7",
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
    logs = _load_json_lines(capsys.readouterr().out)
    assert logs[-1]["event"] == "error"
    assert "referer" in logs[-1]["message"]


def test_validate_requires_ip_for_uid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"

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

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
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
            str(tmp_path / "contract.csv"),
            "--meta",
            str(tmp_path / "meta.json"),
        ]
    )
    assert exit_code == 1
    logs = _load_json_lines(capsys.readouterr().out)
    assert logs[-1]["event"] == "error"
    assert logs[-1]["code"] == "MISSING_MAPPING"
    assert "ip" in logs[-1]["message"]


def test_validate_non_utc_timestamp_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"

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
                "time": "2024-01-01 00:00:00",  # missing timezone
                "user": "user-1",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.1",
                "category": "view",
            }
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--seed",
            "5",
            "validate",
            str(raw_path),
            "--map",
            str(map_path),
            "--out",
            str(tmp_path / "contract.csv"),
            "--meta",
            str(tmp_path / "meta.json"),
        ]
    )
    assert exit_code == 1
    logs = _load_json_lines(capsys.readouterr().out)
    assert logs[-1]["event"] == "error"
    assert logs[-1]["code"] == "INVALID_TIMESTAMP"
    assert "UTC" in logs[-1]["message"]


def _prepare_contract(tmp_path: Path) -> tuple[Path, Path]:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_contract = tmp_path / "meta_contract.json"

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
            # Cluster 1 (short gaps)
            {
                "time": "2024-01-01T00:00:00Z",
                "user": "user-1",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.10",
                "category": "view",
            },
            {
                "time": "2024-01-01T00:00:45Z",
                "user": "user-1",
                "session": "s-1",
                "method": "POST",
                "path": "/search",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.10",
                "category": "search",
            },
            # Second user for fallback behaviour
            {
                "time": "2024-01-01T01:00:00Z",
                "user": "user-2",
                "session": "s-9",
                "method": "GET",
                "path": "/landing",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.11",
                "category": "view",
            },
            {
                "time": "2024-01-01T01:45:00Z",
                "user": "user-2",
                "session": "s-9",
                "method": "GET",
                "path": "/pricing",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.11",
                "category": "view",
            },
            # Large gap to force new session
            {
                "time": "2024-01-01T03:00:45Z",
                "user": "user-1",
                "session": "s-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.10",
                "category": "view",
            },
            {
                "time": "2024-01-01T03:01:10Z",
                "user": "user-1",
                "session": "s-1",
                "method": "POST",
                "path": "/checkout",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.10",
                "category": "convert",
            },
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
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
                str(meta_contract),
            ]
        )
        == 0
    )
    return contract_path, meta_contract


def test_sessionize_applies_threshold_and_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract_path, _ = _prepare_contract(tmp_path)
    sessioned_path = tmp_path / "sessioned.csv"
    meta_path = tmp_path / "meta_session.json"

    exit_code = cli.main(
        [
            "--seed",
            "11",
            "sessionize",
            str(contract_path),
            "--out",
            str(sessioned_path),
            "--meta",
            str(meta_path),
        ]
    )
    assert exit_code == 0
    logs = _load_json_lines(capsys.readouterr().out)
    assert logs[0]["event"] == "start"
    assert logs[-1]["event"] == "complete"
    complete_details = logs[-1]["details"]
    assert complete_details["method"] == "otsu"

    with sessioned_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    expected_uid = _expected_uid("192.0.2.10", "ua", seed=11)
    assert rows[0]["session_id"].startswith(f"{expected_uid}-")
    assert float(rows[1]["delta_t_seconds"]) < 100
    long_gap_row = next(
        row for row in rows if row["timestamp_utc"] == "2024-01-01T03:00:45+00:00"
    )
    assert float(long_gap_row["delta_t_seconds"]) > 1000

    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["seed"] == 11
    assert meta["input_sha256"].isalnum() and len(meta["input_sha256"]) == 64
    threshold = meta["threshold_seconds"]
    assert threshold > float(rows[1]["delta_t_seconds"])  # short gap stays same session
    assert threshold < float(
        long_gap_row["delta_t_seconds"]
    )  # large gap splits session
    assert meta["method"] == "otsu"
    histogram = meta["histogram"]
    assert 32 <= histogram["bin_count"] <= 256
    assert len(histogram["counts"]) == histogram["bin_count"]
    assert complete_details["method"] == meta["method"]
    assert complete_details["input_sha256"] == meta["input_sha256"]
    assert 0.0 <= meta["within_class_variance_ratio"] <= 1.0


def test_sessionize_seed_determinism(tmp_path: Path) -> None:
    contract_path, _ = _prepare_contract(tmp_path)
    sessioned_a = tmp_path / "sessioned_a.csv"
    sessioned_b = tmp_path / "sessioned_b.csv"
    meta_a = tmp_path / "meta_a.json"
    meta_b = tmp_path / "meta_b.json"

    assert (
        cli.main(
            [
                "--seed",
                "23",
                "sessionize",
                str(contract_path),
                "--out",
                str(sessioned_a),
                "--meta",
                str(meta_a),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--seed",
                "23",
                "sessionize",
                str(contract_path),
                "--out",
                str(sessioned_b),
                "--meta",
                str(meta_b),
            ]
        )
        == 0
    )

    assert sessioned_a.read_bytes() == sessioned_b.read_bytes()
    assert meta_a.read_bytes() == meta_b.read_bytes()


def test_validate_seed_determinism(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_a = tmp_path / "contract_a.csv"
    contract_b = tmp_path / "contract_b.csv"
    meta_a = tmp_path / "meta_a.json"
    meta_b = tmp_path / "meta_b.json"

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
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.1",
                "category": "view",
            }
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "--seed",
                "13",
                "validate",
                str(raw_path),
                "--map",
                str(map_path),
                "--out",
                str(contract_a),
                "--meta",
                str(meta_a),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--seed",
                "13",
                "validate",
                str(raw_path),
                "--map",
                str(map_path),
                "--out",
                str(contract_b),
                "--meta",
                str(meta_b),
            ]
        )
        == 0
    )

    assert contract_a.read_bytes() == contract_b.read_bytes()
    assert meta_a.read_bytes() == meta_b.read_bytes()


def test_pipeline_assigns_session_ids_when_source_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_a = tmp_path / "contract_a.csv"
    contract_b = tmp_path / "contract_b.csv"
    session_a = tmp_path / "session_a.csv"
    session_b = tmp_path / "session_b.csv"
    delta_a = tmp_path / "delta_a.csv"
    delta_b = tmp_path / "delta_b.csv"

    _write_csv(
        raw_path,
        fieldnames=[
            "timestamp",
            "user",
            "method",
            "path",
            "referer",
            "user_agent",
            "ip",
            "category",
        ],
        rows=[
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "user": "user-1",
                "method": "GET",
                "path": "/index",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.10",
                "category": "view",
            },
            {
                "timestamp": "2024-01-01T00:10:00Z",
                "user": "user-1",
                "method": "POST",
                "path": "/submit",
                "referer": "https://origin",
                "user_agent": "ua",
                "ip": "192.0.2.10",
                "category": "write",
            },
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: timestamp",
                "uid: user",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        ),
        encoding="utf-8",
    )

    for contract_path, session_path, delta_path in [
        (contract_a, session_a, delta_a),
        (contract_b, session_b, delta_b),
    ]:
        assert (
            cli.main(
                [
                    "--seed",
                    "37",
                    "validate",
                    str(raw_path),
                    "--map",
                    str(map_path),
                    "--out",
                    str(contract_path),
                ]
            )
            == 0
        )

        contract_rows = list(csv.DictReader(contract_path.open("r", encoding="utf-8")))
        assert all(row["session_id"] == "" for row in contract_rows)

        assert (
            cli.main(
                [
                    "--seed",
                    "37",
                    "sessionize",
                    str(contract_path),
                    "--out",
                    str(session_path),
                    "--meta",
                    str(session_path.with_suffix(".meta.json")),
                ]
            )
            == 0
        )
        assert (
            cli.main(
                [
                    "--seed",
                    "37",
                    "deltify",
                    str(session_path),
                    "--out",
                    str(delta_path),
                    "--meta",
                    str(delta_path.with_suffix(".meta.json")),
                ]
            )
            == 0
        )

        session_rows = list(csv.DictReader(session_path.open("r", encoding="utf-8")))
        assert all(row["session_id"] for row in session_rows)
        expected_uid = _expected_uid("192.0.2.10", "ua", seed=37)
        assert all(row["session_id"].startswith(expected_uid) for row in session_rows)

    assert contract_a.read_bytes() == contract_b.read_bytes()
    assert session_a.read_bytes() == session_b.read_bytes()
    assert delta_a.read_bytes() == delta_b.read_bytes()


def test_validate_uid_changes_with_seed(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_a = tmp_path / "contract_a.csv"
    contract_b = tmp_path / "contract_b.csv"

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
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.1",
                "category": "view",
            }
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "--seed",
                "2",
                "validate",
                str(raw_path),
                "--map",
                str(map_path),
                "--out",
                str(contract_a),
                "--meta",
                str(tmp_path / "meta_a.json"),
            ]
        )
        == 0
    )

    assert (
        cli.main(
            [
                "--seed",
                "3",
                "validate",
                str(raw_path),
                "--map",
                str(map_path),
                "--out",
                str(contract_b),
                "--meta",
                str(tmp_path / "meta_b.json"),
            ]
        )
        == 0
    )

    with contract_a.open("r", encoding="utf-8") as handle:
        uid_a = next(csv.DictReader(handle))["uid"]
    with contract_b.open("r", encoding="utf-8") as handle:
        uid_b = next(csv.DictReader(handle))["uid"]

    assert uid_a != uid_b
    assert uid_a == _expected_uid("192.0.2.1", "ua", seed=2)
    assert uid_b == _expected_uid("192.0.2.1", "ua", seed=3)


def test_sessionize_elbow_fallback(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract_uniform.csv"
    meta_contract = tmp_path / "meta_contract_uniform.json"
    raw_path = tmp_path / "raw_uniform.csv"
    map_path = tmp_path / "map_uniform.yaml"

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
                "user": "user-3",
                "session": "sx",
                "method": "GET",
                "path": "/",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.30",
                "category": "view",
            },
            {
                "time": "2024-01-01T00:10:00Z",
                "user": "user-3",
                "session": "sx",
                "method": "GET",
                "path": "/a",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.30",
                "category": "view",
            },
            {
                "time": "2024-01-01T00:20:00Z",
                "user": "user-3",
                "session": "sx",
                "method": "GET",
                "path": "/b",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.30",
                "category": "view",
            },
            {
                "time": "2024-01-01T00:30:00Z",
                "user": "user-3",
                "session": "sx",
                "method": "GET",
                "path": "/c",
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.30",
                "category": "view",
            },
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
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
                str(meta_contract),
            ]
        )
        == 0
    )

    sessioned_path = tmp_path / "session_uniform.csv"
    meta_session = tmp_path / "meta_session_uniform.json"
    assert (
        cli.main(
            [
                "--seed",
                "17",
                "sessionize",
                str(contract_path),
                "--out",
                str(sessioned_path),
                "--meta",
                str(meta_session),
            ]
        )
        == 0
    )

    meta = json.loads(meta_session.read_text(encoding="utf-8"))
    meta = cast(dict[str, Any], meta)
    assert meta["method"] == "elbow"


def _prepare_sessioned(tmp_path: Path) -> tuple[Path, Path]:
    contract_path, _ = _prepare_contract(tmp_path)
    sessioned_path = tmp_path / "sessioned.csv"
    meta_path = tmp_path / "meta_session.json"

    assert (
        cli.main(
            [
                "--seed",
                "31",
                "sessionize",
                str(contract_path),
                "--out",
                str(sessioned_path),
                "--meta",
                str(meta_path),
            ]
        )
        == 0
    )
    return sessioned_path, meta_path


def test_deltify_generates_robust_features(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sessioned_path, _ = _prepare_sessioned(tmp_path)
    deltified_path = tmp_path / "deltified.csv"
    meta_path = tmp_path / "meta_dt.json"

    exit_code = cli.main(
        [
            "--seed",
            "31",
            "deltify",
            str(sessioned_path),
            "--out",
            str(deltified_path),
            "--meta",
            str(meta_path),
        ]
    )
    assert exit_code == 0
    logs = _load_json_lines(capsys.readouterr().out)
    assert logs[0]["event"] == "start"
    assert logs[-1]["event"] == "complete"
    complete_details = logs[-1]["details"]

    with deltified_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert "robust_z" in rows[0]
    assert "robust_z_clipped" in rows[0]
    assert "burst_ratio" in rows[0]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = cast(dict[str, Any], meta)
    assert meta["seed"] == 31
    assert meta["clip_range"] == [-5, 5]
    assert "global_median" in meta
    expected_uid = _expected_uid("192.0.2.11", "ua", seed=11)
    assert expected_uid in meta["fallback_users"]
    assert meta["group"] == "uid"
    assert complete_details["input_sha256"] == meta["input_sha256"]


def test_deltify_seed_determinism(tmp_path: Path) -> None:
    sessioned_path, _ = _prepare_sessioned(tmp_path)
    meta_a = tmp_path / "meta_dt_a.json"
    meta_b = tmp_path / "meta_dt_b.json"
    out_a = tmp_path / "deltified_a.csv"
    out_b = tmp_path / "deltified_b.csv"

    assert (
        cli.main(
            [
                "--seed",
                "19",
                "deltify",
                str(sessioned_path),
                "--out",
                str(out_a),
                "--meta",
                str(meta_a),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--seed",
                "19",
                "deltify",
                str(sessioned_path),
                "--out",
                str(out_b),
                "--meta",
                str(meta_b),
            ]
        )
        == 0
    )

    assert out_a.read_bytes() == out_b.read_bytes()
    assert meta_a.read_bytes() == meta_b.read_bytes()


def test_missing_seed_triggers_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    map_path = tmp_path / "map.yaml"
    contract_path = tmp_path / "contract.csv"
    meta_path = tmp_path / "meta_contract.json"

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
                "referer": "-",
                "user_agent": "ua",
                "ip": "192.0.2.1",
                "category": "view",
            }
        ],
    )

    map_path.write_text(
        "\n".join(
            [
                "timestamp_utc: time",
                "uid: user",
                "session_id: session",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "ip: ip",
                "op_category: category",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
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
    logs = _load_json_lines(capsys.readouterr().out)
    assert logs[-1]["event"] == "error"
    assert logs[-1]["code"] == "MISSING_SEED"
