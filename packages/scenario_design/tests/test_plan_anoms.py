import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))


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


def _prepare_stats(tmp_path: Path) -> Path:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,2.5",
        "2024-01-01T00:00:04+00:00,u1,s1,POST,/read,,ua,10.0.0.1,READ,1.5",
        "2024-01-01T00:00:09+00:00,u1,s1,POST,/update,,ua,10.0.0.1,UPDATE,3.5",
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
            "17",
        ]
    )
    assert fit_result.returncode == 0, fit_result.stderr.decode()
    return stats_path


def test_plan_cli_accepts_multiple_anomaly_options(tmp_path: Path) -> None:
    stats_path = _prepare_stats(tmp_path)

    spec_path = tmp_path / "scenario_spec.json"
    plan_seed = 31
    plan_result = run_cli(
        [
            "plan",
            "--stats",
            str(stats_path),
            "--out",
            str(spec_path),
            "--seed",
            str(plan_seed),
            "--anom",
            "time(mode=propagate,p=0.02,scale=3.0)",
            "--anom",
            "order(p=0.01)",
            "--anom",
            "unauth(p=0.005)",
            "--anom",
            "token_replay(p=0.004)",
        ]
    )
    assert plan_result.returncode == 0, plan_result.stderr.decode()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["seed"] == plan_seed
    assert spec["anoms"] == [
        {"type": "time", "mode": "propagate", "p": 0.02, "scale": 3.0},
        {"type": "order", "p": 0.01},
        {"type": "unauth", "p": 0.005},
        {"type": "token_replay", "p": 0.004},
    ]


def test_plan_cli_is_deterministic_with_time_delta(tmp_path: Path) -> None:
    stats_path = _prepare_stats(tmp_path)

    first_spec = tmp_path / "spec1.json"
    second_spec = tmp_path / "spec2.json"
    seed = 73
    args = [
        "plan",
        "--stats",
        str(stats_path),
        "--out",
        str(first_spec),
        "--seed",
        str(seed),
        "--anom",
        "time(mode=local,p=0.03,delta=-1.5)",
    ]
    result_one = run_cli(args)
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
            "--anom",
            "time(mode=local,p=0.03,delta=-1.5)",
        ]
    )
    assert result_two.returncode == 0, result_two.stderr.decode()

    assert first_spec.read_bytes() == second_spec.read_bytes()

    spec = json.loads(first_spec.read_text(encoding="utf-8"))
    assert spec["anoms"] == [
        {"type": "time", "mode": "local", "p": 0.03, "delta": -1.5}
    ]


def test_plan_cli_rejects_invalid_time_anomaly(tmp_path: Path) -> None:
    stats_path = _prepare_stats(tmp_path)

    spec_path = tmp_path / "scenario_spec.json"
    result = run_cli(
        [
            "plan",
            "--stats",
            str(stats_path),
            "--out",
            str(spec_path),
            "--seed",
            "5",
            "--anom",
            "time(p=0.2,scale=2.0)",
        ]
    )
    assert result.returncode == 2

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[-1]["event"] == "error"
    assert stdout_lines[-1]["error_code"] == "SCENARIO_DESIGN_ARGUMENT_ERROR"
    assert "mode" in stdout_lines[-1]["message"].lower()
    assert not spec_path.exists()
