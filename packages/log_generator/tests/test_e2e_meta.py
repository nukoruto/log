import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from click.testing import CliRunner

from log_generator.cli import cli


def _write_minimal_spec(base_path: Path) -> Path:
    spec = {
        "algo_version": "0.1.0",
        "t0": "1970-01-01T00:00:00Z",
        "users": [
            {
                "uid": "user-1",
                "session_id": "sess-1",
                "initial_op": "login",
                "steps": 2,
            }
        ],
        "ops": {
            "login": {
                "method": "POST",
                "path": "/login",
                "referer": "https://example.test/",
                "user_agent": "SpecAgent/1.0",
                "ip": "192.0.2.1",
                "op_category": "auth",
                "transitions": [
                    {"op": "dashboard", "prob": 1.0},
                ],
                "dt_distribution": {
                    "type": "piecewise",
                    "cdf": [
                        {"p": 0.0, "seconds": 1.0},
                        {"p": 1.0, "seconds": 1.0},
                    ],
                },
            },
            "dashboard": {
                "method": "GET",
                "path": "/dashboard",
                "referer": "https://example.test/login",
                "user_agent": "SpecAgent/1.0",
                "ip": "192.0.2.1",
                "op_category": "browse",
                "transitions": [
                    {"op": "dashboard", "prob": 1.0},
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
        "anoms": [],
    }
    spec_path = base_path / "scenario_spec.json"
    spec_path.write_text(json.dumps(spec))
    return spec_path


def test_run_meta_serialization_is_canonical(tmp_path: Path) -> None:
    spec_path = _write_minimal_spec(tmp_path)
    meta_path = tmp_path / "run_meta.json"
    audit_path = tmp_path / "audit.jsonl"
    normal_path = tmp_path / "normal.csv"
    anom_path = tmp_path / "anom.csv"

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

    assert result.exit_code == 0, result.output

    spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    expected_payload = {
        "algo_version": "0.1.0",
        "seed": 42,
        "spec_sha256": spec_sha,
    }
    expected_bytes = (
        json.dumps(expected_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

    meta_bytes = meta_path.read_bytes()
    assert meta_bytes == expected_bytes

    meta_second = tmp_path / "run_meta_second.json"
    audit_second = tmp_path / "audit_second.jsonl"
    normal_second = tmp_path / "normal_second.csv"
    anom_second = tmp_path / "anom_second.csv"

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
    assert meta_second.read_bytes() == expected_bytes
