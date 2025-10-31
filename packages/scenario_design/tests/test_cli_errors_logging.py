import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def run_cli(
    args: List[str], env: Dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    base_env = os.environ.copy()
    package_root = Path(__file__).resolve().parents[1] / "src"
    pythonpath = str(package_root)
    if env is None:
        env = {}
    base_env["PYTHONPATH"] = pythonpath + os.pathsep + base_env.get("PYTHONPATH", "")
    completed = subprocess.run(
        ["python", "-m", "scenario_design.cli", *args],
        env={**base_env, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed


def parse_json_lines(output: bytes) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in output.decode().strip().splitlines()
        if line.strip()
    ]


def test_contract_error_logs_include_standard_fields(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,__unknown__,2.0",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    stats_path = tmp_path / "stats.pkl"

    result = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path),
            "--seed",
            "101",
        ]
    )
    assert result.returncode == 1

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[0]["event"] == "start"
    error_record = stdout_lines[-1]
    assert error_record["event"] == "error"
    assert error_record["command"] == "fit"
    assert error_record["error_code"] == "SCENARIO_DESIGN_CONTRACT_ERROR"
    assert error_record["seed"] == 101
    assert error_record["input"] == str(csv_path)
    assert error_record["output"] == str(stats_path)
    assert error_record["input_sha256"]


def test_schema_violation_uses_contract_error_code(tmp_path: Path) -> None:
    stats_path = tmp_path / "invalid_stats.pkl"
    invalid_stats = {
        "categories": ["AUTH"],
        "pi": {"AUTH": 1.0},
        "A": {"AUTH": {"AUTH": 1.0}},
        "lognorm": {"mu": {"AUTH": float("nan")}, "sigma": {"AUTH": 1.0}},
        "n_events": 1,
        "n_sessions": 1,
    }
    stats_path.write_bytes(pickle.dumps(invalid_stats))
    spec_path = tmp_path / "spec.json"

    result = run_cli(
        [
            "plan",
            "--stats",
            str(stats_path),
            "--out",
            str(spec_path),
            "--seed",
            "202",
        ]
    )
    assert result.returncode == 1

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[0]["event"] == "start"
    error_record = stdout_lines[-1]
    assert error_record["event"] == "error"
    assert error_record["command"] == "plan"
    assert error_record["error_code"] == "SCENARIO_DESIGN_CONTRACT_ERROR"
    assert error_record["seed"] == 202
    assert error_record["input"] == str(stats_path)
    assert error_record["output"] == str(spec_path)
    assert error_record["input_sha256"] == stdout_lines[0]["input_sha256"]


def test_missing_seed_reports_argument_error(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,1.0",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    stats_path = tmp_path / "stats.pkl"

    result = run_cli(["fit", str(csv_path), "--out", str(stats_path)])
    assert result.returncode == 2

    stdout_lines = parse_json_lines(result.stdout)
    error_record = stdout_lines[-1]
    assert error_record["event"] == "error"
    assert error_record["command"] == "fit"
    assert error_record["error_code"] == "SCENARIO_DESIGN_ARGUMENT_ERROR"
    assert error_record["seed"] is None
    assert error_record["input"] is None
    assert error_record["output"] is None
    assert "input_sha256" in error_record
