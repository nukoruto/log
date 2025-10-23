import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from scenario_design.schema import validate_scenario_spec  # noqa: E402


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


def test_plan_generates_schema_compliant_normal_spec(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,2.0",
        "2024-01-01T00:00:05+00:00,u1,s1,POST,/read,,ua,10.0.0.1,READ,1.0",
        "2024-01-01T00:00:10+00:00,u1,s1,POST,/read,,ua,10.0.0.1,READ,4.0",
        "2024-01-01T00:00:20+00:00,u1,s1,POST,/update,,ua,10.0.0.1,UPDATE,3.0",
        "2024-01-02T00:00:00+00:00,u2,s2,GET,/auth,,ua,10.0.0.2,AUTH,5.0",
        "2024-01-02T00:00:03+00:00,u2,s2,POST,/read,,ua,10.0.0.2,READ,2.0",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    stats_path = tmp_path / "stats.pkl"
    fit_seed = 7
    fit_result = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path),
            "--seed",
            str(fit_seed),
        ]
    )
    assert fit_result.returncode == 0, fit_result.stderr.decode()

    plan_seed = 42
    spec_path = tmp_path / "scenario_spec.json"
    plan_result = run_cli(
        [
            "plan",
            "--stats",
            str(stats_path),
            "--out",
            str(spec_path),
            "--seed",
            str(plan_seed),
        ]
    )
    assert plan_result.returncode == 0, plan_result.stderr.decode()

    stdout_lines = parse_json_lines(plan_result.stdout)
    assert stdout_lines[0]["event"] == "start"
    assert stdout_lines[-1]["event"] == "complete"
    assert stdout_lines[0]["command"] == "plan"
    assert stdout_lines[-1]["command"] == "plan"
    assert stdout_lines[0]["seed"] == plan_seed
    assert stdout_lines[-1]["seed"] == plan_seed
    assert stdout_lines[0]["input"] == str(stats_path)
    assert stdout_lines[-1]["input"] == str(stats_path)
    assert stdout_lines[0]["output"] == str(spec_path)
    assert stdout_lines[-1]["output"] == str(spec_path)
    assert stdout_lines[0]["input_sha256"] == stdout_lines[-1]["input_sha256"]

    stats = pickle.loads(stats_path.read_bytes())
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["length"] == stats["n_events"]
    assert spec["users"] == stats["n_sessions"]
    assert spec["pi"] == stats["pi"]
    assert spec["A"] == stats["A"]

    expected_sigma = {
        category: (value if value > 0.0 else pytest.approx(1e-9, rel=1e-6))
        for category, value in stats["lognorm"]["sigma"].items()
    }
    assert spec["dt"]["lognorm"]["mu"] == stats["lognorm"]["mu"]
    for category, sigma_value in spec["dt"]["lognorm"]["sigma"].items():
        assert sigma_value == expected_sigma[category]

    assert spec["anoms"] == []
    assert spec["seed"] == plan_seed

    # Schema validation must pass
    validate_scenario_spec(spec)


def test_plan_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,1.0",
        "2024-01-01T00:00:02+00:00,u1,s1,POST,/read,,ua,10.0.0.1,READ,1.5",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    stats_path = tmp_path / "stats.pkl"
    fit_result = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path),
            "--seed",
            "11",
        ]
    )
    assert fit_result.returncode == 0, fit_result.stderr.decode()

    first_spec = tmp_path / "spec1.json"
    second_spec = tmp_path / "spec2.json"
    seed = 19

    result_one = run_cli(
        [
            "plan",
            "--stats",
            str(stats_path),
            "--out",
            str(first_spec),
            "--seed",
            str(seed),
        ]
    )
    assert result_one.returncode == 0, result_one.stderr.decode()

    result_two = run_cli(
        [
            "plan",
            "--stats",
            str(stats_path),
            "--out",
            str(second_spec),
            "--seed",
            str(seed),
        ]
    )
    assert result_two.returncode == 0, result_two.stderr.decode()

    assert first_spec.read_bytes() == second_spec.read_bytes()


def test_plan_requires_seed(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,1.0",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    stats_path = tmp_path / "stats.pkl"
    fit_result = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path),
            "--seed",
            "3",
        ]
    )
    assert fit_result.returncode == 0, fit_result.stderr.decode()

    spec_path = tmp_path / "scenario_spec.json"
    result = run_cli(
        [
            "plan",
            "--stats",
            str(stats_path),
            "--out",
            str(spec_path),
        ]
    )
    assert result.returncode != 0

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[-1]["event"] == "error"
    assert stdout_lines[-1]["error_code"] == "SCENARIO_DESIGN_ARGUMENT_ERROR"
    assert "seed" in stdout_lines[-1]["message"].lower()
    assert stdout_lines[-1]["seed"] is None
    assert not spec_path.exists()
