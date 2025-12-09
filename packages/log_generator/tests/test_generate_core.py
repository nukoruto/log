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
        "length": 5,
        "users": 2,
        "pi": {"AUTH": 1.0},
        "A": {"AUTH": {"READ": 1.0}, "READ": {"READ": 1.0}},
        "dt": {
            "lognorm": {
                "mu": {"AUTH": 0.0, "READ": 0.6931471805599453},
                "sigma": {"AUTH": 1e-9, "READ": 1e-9},
            }
        },
        "anoms": [
            {
                "type": "time",
                "mode": "propagate",
                "p": 1.0,
                "scale": 2.0,
                "op": "READ",
            }
        ],
        "seed": 7,
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

    # Operations come from the pi/A categories.
    spec = json.loads(spec_path.read_text())
    categories = set(spec["pi"].keys()) | set(spec["A"].keys())
    for targets in spec["A"].values():
        categories.update(targets.keys())
    assert {row["op_category"] for row in rows}.issubset(categories)

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

    assert meta_data["seed"] == 42
    assert meta_data["spec_sha256"] == expected_sha
    assert "algo_version" in meta_data

    # Audit log contains structured entries for injected anomalies.
    audit_entries = [
        json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()
    ]
    assert audit_entries, "audit log must contain anomaly entries"
    assert all(entry.get("seed") == 42 for entry in audit_entries)
    assert {entry["type"] for entry in audit_entries} == {"time"}
    assert all(isinstance(entry.get("record_index"), int) for entry in audit_entries)

    # Ensure at least one anomaly altered the timeline.
    differences = [
        index
        for index, (normal_row, anom_row) in enumerate(zip(rows, anom_rows))
        if normal_row != anom_row
    ]
    assert differences, "anom.csv must differ from normal.csv when anomalies are injected"


def test_cli_respects_t0_override(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path)
    normal_path = tmp_path / "normal.csv"
    anom_path = tmp_path / "anom.csv"
    audit_path = tmp_path / "audit.jsonl"
    meta_path = tmp_path / "run_meta.json"
    runner = CliRunner()

    custom_t0 = "2024-02-03T04:05:06Z"
    result = runner.invoke(
        cli,
        [
            "run",
            "--spec",
            str(spec_path),
            "--t0",
            custom_t0,
            "--seed",
            "11",
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
    rows = load_rows(normal_path)
    anom_rows = load_rows(anom_path)
    assert rows[0]["timestamp_utc"] == custom_t0
    assert anom_rows[0]["timestamp_utc"] == custom_t0


def test_cli_rejects_non_utc_t0(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path)
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
            "--t0",
            "2024-02-03T04:05:06+09:00",
            "--seed",
            "11",
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
    assert any("UTC" in entry.get("message", "") for entry in logs)


def test_cli_logs_error_metadata_on_spec_failure(tmp_path: Path) -> None:
    spec_path = tmp_path / "invalid_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "length": 4,
                "users": 1,
                "pi": {"AUTH": 1.0},
                "A": {"AUTH": {"READ": 1.0}},
                # Missing dt block
                "anoms": [],
                "seed": 99,
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
    assert "dt" in error_log["message"]

    assert not normal_path.exists()
    assert not anom_path.exists()
    assert not audit_path.exists()
    assert not meta_path.exists()
