import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest


def run_cli(
    args: list[str], env: Dict[str, str] | None = None
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


def test_fit_estimates_markov_and_lognormal(tmp_path: Path) -> None:
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

    seed = 7
    result = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path),
            "--seed",
            str(seed),
        ]
    )
    assert result.returncode == 0, result.stderr.decode()

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[0]["event"] == "start"
    assert stdout_lines[-1]["event"] == "complete"
    assert stdout_lines[0]["seed"] == seed
    assert stdout_lines[-1]["seed"] == seed
    assert stdout_lines[0]["input"] == str(csv_path)
    assert stdout_lines[-1]["input"] == str(csv_path)
    assert stdout_lines[0]["output"] == str(stats_path)
    assert stdout_lines[-1]["output"] == str(stats_path)
    assert stdout_lines[0]["input_sha256"] == stdout_lines[-1]["input_sha256"]

    stats = pickle.loads(stats_path.read_bytes())
    assert set(stats["categories"]) == {"AUTH", "READ", "UPDATE"}
    assert stats["seed"] == seed
    assert stats["input_sha256"] == stdout_lines[-1]["input_sha256"]

    pi = stats["pi"]
    assert pytest.approx(pi["AUTH"], rel=1e-6) == 7 / 9
    assert pytest.approx(pi["READ"], rel=1e-6) == 1 / 9
    assert pytest.approx(pi["UPDATE"], rel=1e-6) == 1 / 9

    transitions: Dict[str, Dict[str, float]] = stats["A"]
    assert pytest.approx(transitions["AUTH"]["READ"], rel=1e-6) == 7 / 9
    assert pytest.approx(transitions["AUTH"]["AUTH"], rel=1e-6) == 1 / 9
    assert pytest.approx(transitions["READ"]["READ"], rel=1e-6) == pytest.approx(
        4 / 9, rel=1e-6
    )
    assert pytest.approx(transitions["READ"]["UPDATE"], rel=1e-6) == pytest.approx(
        4 / 9, rel=1e-6
    )

    mu = stats["lognorm"]["mu"]
    sigma = stats["lognorm"]["sigma"]
    assert pytest.approx(mu["AUTH"], rel=1e-6) == pytest.approx(1.1512925465, rel=1e-6)
    assert pytest.approx(sigma["AUTH"], rel=1e-6) == pytest.approx(
        0.4581453659, rel=1e-6
    )
    assert pytest.approx(mu["UPDATE"], rel=1e-6) == pytest.approx(
        1.0986122887, rel=1e-6
    )
    assert sigma["UPDATE"] == 0.0


def test_fit_rejects_unknown_operations(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,__unknown__,2.0",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    stats_path = tmp_path / "stats.pkl"

    seed = 11
    result = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path),
            "--seed",
            str(seed),
        ]
    )
    assert result.returncode != 0

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[0]["event"] == "start"
    assert stdout_lines[0]["seed"] == seed
    assert stdout_lines[0]["input"] == str(csv_path)
    assert stdout_lines[0]["output"] == str(stats_path)
    assert stdout_lines[0]["input_sha256"]
    assert stdout_lines[-1]["event"] == "error"
    assert stdout_lines[-1]["seed"] == seed
    assert stdout_lines[-1]["error_code"] == "SCENARIO_DESIGN_CONTRACT_ERROR"
    assert "unknown op" in stdout_lines[-1]["message"].lower()
    assert stdout_lines[-1]["input_sha256"]
    assert not stats_path.exists()


@pytest.mark.parametrize("delta_value", ["NaN", "Infinity", "-1.0", "0.0"])
def test_fit_rejects_non_finite_delta_t(tmp_path: Path, delta_value: str) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        f"2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,{delta_value}",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    stats_path = tmp_path / "stats.pkl"

    seed = 17
    result = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path),
            "--seed",
            str(seed),
        ]
    )
    assert result.returncode != 0

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[0]["event"] == "start"
    assert stdout_lines[-1]["event"] == "error"
    assert stdout_lines[-1]["error_code"] == "SCENARIO_DESIGN_CONTRACT_ERROR"
    message = stdout_lines[-1]["message"].lower()
    assert "delta_t" in message
    assert "finite positive" in message
    assert "row 2" in message
    assert delta_value.lower() in message
    assert not stats_path.exists()


def test_fit_requires_seed(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,2.0",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    stats_path = tmp_path / "stats.pkl"

    result = run_cli(["fit", str(csv_path), "--out", str(stats_path)])
    assert result.returncode != 0

    stdout_lines = parse_json_lines(result.stdout)
    assert stdout_lines[-1]["event"] == "error"
    assert stdout_lines[-1]["error_code"] == "SCENARIO_DESIGN_ARGUMENT_ERROR"
    assert "seed" in stdout_lines[-1]["message"].lower()
    assert stdout_lines[-1]["seed"] is None
    assert "input_sha256" in stdout_lines[-1]
    assert not stats_path.exists()


def test_fit_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    rows = [
        "timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category,delta_t",
        "2024-01-01T00:00:00+00:00,u1,s1,GET,/auth,,ua,10.0.0.1,AUTH,2.0",
        "2024-01-01T00:00:05+00:00,u1,s1,POST,/read,,ua,10.0.0.1,READ,1.0",
    ]
    csv_path = tmp_path / "deltified.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    stats_path_one = tmp_path / "stats1.pkl"
    stats_path_two = tmp_path / "stats2.pkl"
    seed = 13

    result_one = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path_one),
            "--seed",
            str(seed),
        ]
    )
    assert result_one.returncode == 0, result_one.stderr.decode()

    result_two = run_cli(
        [
            "fit",
            str(csv_path),
            "--out",
            str(stats_path_two),
            "--seed",
            str(seed),
        ]
    )
    assert result_two.returncode == 0, result_two.stderr.decode()

    assert stats_path_one.read_bytes() == stats_path_two.read_bytes()
