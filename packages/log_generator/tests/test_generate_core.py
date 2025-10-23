from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from click.testing import CliRunner

from log_generator.cli import cli


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


def write_spec(tmp_path: Path) -> Path:
    spec = {
        "algo_version": "0.1.0",
        "t0": "1970-01-01T00:00:00Z",
        "users": [
            {
                "uid": "user-1",
                "session_id": "sess-1",
                "initial_op": "login",
                "steps": 3,
            },
            {
                "uid": "user-2",
                "session_id": "sess-2",
                "initial_op": "login",
                "steps": 2,
            },
        ],
        "ops": {
            "login": {
                "method": "POST",
                "path": "/login",
                "referer": "https://example.test/",
                "user_agent": "SpecAgent/1.0",
                "ip": "192.0.2.10",
                "op_category": "auth",
                "transitions": [
                    {"op": "dashboard", "prob": 1.0},
                ],
                "dt_distribution": {
                    "type": "piecewise",
                    "cdf": [
                        {"p": 0.0, "seconds": 3.0},
                        {"p": 0.5, "seconds": 5.0},
                        {"p": 1.0, "seconds": 7.0},
                    ],
                },
            },
            "dashboard": {
                "method": "GET",
                "path": "/dashboard",
                "referer": "https://example.test/login",
                "user_agent": "SpecAgent/1.0",
                "ip": "198.51.100.5",
                "op_category": "browse",
                "transitions": [
                    {"op": "logout", "prob": 1.0},
                ],
                "dt_distribution": {
                    "type": "piecewise",
                    "cdf": [
                        {"p": 0.0, "seconds": 11.0},
                        {"p": 1.0, "seconds": 11.0},
                    ],
                },
            },
            "logout": {
                "method": "POST",
                "path": "/logout",
                "referer": "https://example.test/dashboard",
                "user_agent": "SpecAgent/1.0",
                "ip": "203.0.113.2",
                "op_category": "auth",
                "transitions": [
                    {"op": "logout", "prob": 1.0},
                ],
                "dt_distribution": {
                    "type": "piecewise",
                    "cdf": [
                        {"p": 0.0, "seconds": 2.0},
                        {"p": 1.0, "seconds": 2.0},
                    ],
                },
            },
        },
        "anoms": [
            {
                "type": "time",
                "mode": "propagate",
                "p": 1.0,
                "scale": 3.0,
                "op": "dashboard",
            },
            {"type": "order", "p": 1.0, "op": "dashboard"},
        ],
    }
    path = tmp_path / "scenario_spec.json"
    path.write_text(json.dumps(spec))
    return path


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    return rows


def test_cli_generates_contract_outputs_with_anomalies(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path)
    normal_path = tmp_path / "normal.csv"
    anom_path = tmp_path / "anom.csv"
    audit_path = tmp_path / "audit.jsonl"

    runner = CliRunner()
    meta_path = tmp_path / "run_meta.json"
    result = runner.invoke(
        cli,
        [
            "run",
            "--spec",
            str(spec_path),
            "--seed",
            "42",
            "--normal",
            str(normal_path),
            "--anom",
            str(anom_path),
            "--audit",
            str(audit_path),
            "--meta",
            str(meta_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert normal_path.exists()
    assert anom_path.exists()
    assert audit_path.exists()
    assert meta_path.exists()

    rows = load_rows(normal_path)
    anom_rows = load_rows(anom_path)
    assert len(rows) == 5
    assert len(anom_rows) == len(rows)

    assert rows[0]["timestamp_utc"].endswith("Z")
    assert anom_rows[0]["timestamp_utc"].endswith("Z")

    for column in CONTRACT_COLUMNS:
        assert column in rows[0]
        assert column in anom_rows[0]

    timestamps = [
        datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
        for row in rows
    ]
    assert timestamps == sorted(timestamps)

    anom_timestamps = [
        datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
        for row in anom_rows
    ]
    assert anom_timestamps == sorted(anom_timestamps)

    # Methods and paths correspond to the spec definitions for each op.
    ops = {
        name: details
        for name, details in json.loads(spec_path.read_text())["ops"].items()
    }
    for row in rows:
        matching = [
            name
            for name, details in ops.items()
            if details["method"] == row["method"]
            and details["path"] == row["path"]
            and details["op_category"] == row["op_category"]
        ]
        assert matching, f"row does not match any op definition: {row}"

    # Determinism: re-run and ensure byte-identical output.
    normal_second = tmp_path / "normal-second.csv"
    anom_second = tmp_path / "anom-second.csv"
    meta_second = tmp_path / "run-meta-second.json"
    audit_second = tmp_path / "audit-second.jsonl"
    second_result = runner.invoke(
        cli,
        [
            "run",
            "--spec",
            str(spec_path),
            "--seed",
            "42",
            "--normal",
            str(normal_second),
            "--anom",
            str(anom_second),
            "--audit",
            str(audit_second),
            "--meta",
            str(meta_second),
        ],
    )
    assert second_result.exit_code == 0, second_result.output
    assert normal_path.read_bytes() == normal_second.read_bytes()
    assert anom_path.read_bytes() == anom_second.read_bytes()
    assert audit_path.read_bytes() == audit_second.read_bytes()
    assert meta_second.exists()

    # UTC order and increments greater than or equal to zero.
    deltas = [
        (timestamps[index] - timestamps[index - 1]).total_seconds()
        for index in range(1, len(timestamps))
    ]
    for delta in deltas:
        assert delta >= 0

    anom_deltas = [
        (anom_timestamps[index] - anom_timestamps[index - 1]).total_seconds()
        for index in range(1, len(anom_timestamps))
    ]
    for delta in anom_deltas:
        assert delta >= 0

    meta_data = json.loads(meta_path.read_text())
    spec_bytes = spec_path.read_bytes()
    expected_sha = hashlib.sha256(spec_bytes).hexdigest()

    assert meta_data == {
        "seed": 42,
        "algo_version": "0.1.0",
        "spec_sha256": expected_sha,
    }

    # Audit log contains structured entries for injected anomalies.
    audit_entries = [
        json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()
    ]
    assert audit_entries, "audit log must contain anomaly entries"
    assert all(entry.get("seed") == 42 for entry in audit_entries)
    assert {entry["type"] for entry in audit_entries} >= {"time", "order"}
    assert all(isinstance(entry.get("record_index"), int) for entry in audit_entries)

    # Ensure at least one anomaly altered the timeline or sequence.
    differences = [
        index
        for index, (normal_row, anom_row) in enumerate(zip(rows, anom_rows))
        if normal_row != anom_row
    ]
    assert (
        differences
    ), "anom.csv must differ from normal.csv when anomalies are injected"


def test_cli_logs_error_metadata_on_spec_failure(tmp_path: Path) -> None:
    spec_path = tmp_path / "invalid_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "t0": "1970-01-01T00:00:00Z",
                "users": [
                    {
                        "uid": "user-1",
                        "session_id": "sess-1",
                        "initial_op": "login",
                        "steps": 1,
                    }
                ],
                "ops": {
                    "login": {
                        "method": "POST",
                        "path": "/login",
                        "referer": "https://example.test/",
                        "user_agent": "SpecAgent/1.0",
                        "ip": "192.0.2.10",
                        "op_category": "auth",
                        "transitions": [
                            {"op": "login", "prob": 1.0},
                        ],
                        "dt_distribution": {
                            "type": "piecewise",
                            "cdf": [
                                {"p": 0.0, "seconds": 1.0},
                                {"p": 1.0, "seconds": 1.0},
                            ],
                        },
                    }
                },
                "anoms": [],
            }
        )
    )

    normal_path = tmp_path / "normal.csv"
    anom_path = tmp_path / "anom.csv"
    audit_path = tmp_path / "audit.jsonl"
    meta_path = tmp_path / "run_meta.json"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--spec",
            str(spec_path),
            "--seed",
            "42",
            "--normal",
            str(normal_path),
            "--anom",
            str(anom_path),
            "--audit",
            str(audit_path),
            "--meta",
            str(meta_path),
        ],
    )

    assert result.exit_code == 1

    logs = [
        json.loads(line)
        for line in result.output.splitlines()
        if line.strip().startswith("{")
    ]
    assert logs, result.output
    assert logs[0]["event"] == "start"
    error_log = logs[-1]
    assert error_log["event"] == "error"
    assert error_log["seed"] == 42
    assert error_log["spec"] == str(spec_path)
    expected_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert error_log["spec_sha256"] == expected_sha
    assert error_log["normal"] == str(normal_path)
    assert error_log["anom"] == str(anom_path)
    assert error_log["audit"] == str(audit_path)
    assert error_log["meta"] == str(meta_path)
    assert "algo_version" in error_log["message"]

    assert not normal_path.exists()
    assert not anom_path.exists()
    assert not audit_path.exists()
    assert not meta_path.exists()
