import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "packages" / "log_generator" / "src"


@pytest.fixture()
def scenario_spec(tmp_path: Path) -> Path:
    spec = {
        "length": 6,
        "t0": "2024-01-01T00:00:00Z",
        "users": {"count": 3},
        "uid_hmac_key": "tests-secret-key",
        "pi": {"AUTH": 1.0},
        "A": {"AUTH": {"AUTH": 1.0}},
        "dt": {
            "lognorm": {
                "mu": {"AUTH": 0.0},
                "sigma": {"AUTH": 0.15},
            },
            "min_seconds": 0.1,
        },
        "catalog": {
            "AUTH": {
                "method": "POST",
                "path": "/login",
                "referer": "https://example.test/landing",
                "user_agent": "pytest-agent/1.0",
                "ip": "198.51.100.10",
            }
        },
        "algo_version": "0.1.0",
    }
    spec_path = tmp_path / "scenario_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec_path


def _run_cli(spec_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    cmd = [
        sys.executable,
        "-m",
        "log_generator.cli",
        "run",
        "--spec",
        str(spec_path),
        "--seed",
        "42",
        "--normal",
        str(output_path),
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def test_cli_generates_contract_csv_and_logs_json(
    tmp_path: Path, scenario_spec: Path
) -> None:
    normal_path = tmp_path / "normal.csv"
    result = _run_cli(scenario_spec, normal_path)

    log_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(log_lines) >= 2, "expected start/end JSON logs"
    for raw in log_lines:
        payload = json.loads(raw)
        assert payload["seed"] == 42
        assert payload["event"].startswith("log_generator.run")

    data = normal_path.read_bytes()
    assert data, "normal.csv should not be empty"

    rows = list(csv.DictReader(normal_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 6
    expected_columns = [
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
    assert list(rows[0].keys()) == expected_columns

    timestamps = []
    uid_pattern = re.compile(r"^[0-9a-f]{64}$")
    for row in rows:
        ts_raw = row["timestamp_utc"]
        assert ts_raw.endswith("Z")
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        assert ts.tzinfo == timezone.utc
        timestamps.append(ts)
        assert uid_pattern.match(row["uid"])
        assert uid_pattern.match(row["session_id"])
        assert row["method"] == "POST"
        assert row["path"] == "/login"
        assert row["referer"] == "https://example.test/landing"
        assert row["user_agent"] == "pytest-agent/1.0"
        assert row["ip"] == "198.51.100.10"
        assert row["op_category"] == "AUTH"

    for before, after in zip(timestamps, timestamps[1:]):
        assert after > before, "timestamps must increase strictly"


def test_cli_is_deterministic_for_same_seed(
    tmp_path: Path, scenario_spec: Path
) -> None:
    first = tmp_path / "normal_first.csv"
    second = tmp_path / "normal_second.csv"

    result_one = _run_cli(scenario_spec, first)
    result_two = _run_cli(scenario_spec, second)

    assert first.read_bytes() == second.read_bytes()

    log_one = [
        json.loads(line)
        for line in result_one.stdout.strip().splitlines()
        if line.strip()
    ]
    log_two = [
        json.loads(line)
        for line in result_two.stdout.strip().splitlines()
        if line.strip()
    ]
    assert log_one[-1]["spec_sha256"] == log_two[-1]["spec_sha256"]
    assert log_one[-1]["records"] == log_two[-1]["records"]
