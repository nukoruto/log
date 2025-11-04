"""End-to-end integration tests for the models-lstm CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("torch")


def _write_contract_csv(path: Path) -> None:
    columns = [
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
    rows = [
        [
            "2023-01-01T00:00:00Z",
            "u1",
            "s1",
            "GET",
            "/a",
            "-",
            "ua",
            "1.1.1.1",
            "cat_a",
        ],
        [
            "2023-01-01T00:01:00Z",
            "u1",
            "s1",
            "POST",
            "/b",
            "-",
            "ua",
            "1.1.1.1",
            "cat_b",
        ],
        [
            "2023-01-02T00:00:00Z",
            "u2",
            "s2",
            "GET",
            "/c",
            "-",
            "ua",
            "2.2.2.2",
            "cat_a",
        ],
        [
            "2023-01-02T00:02:00Z",
            "u2",
            "s2",
            "POST",
            "/d",
            "-",
            "ua",
            "2.2.2.2",
            "cat_b",
        ],
    ]
    lines = [",".join(columns)]
    lines.extend(",".join(str(field) for field in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _run_cli_command(arguments: Iterable[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC_ROOT}{os.pathsep}{existing}" if existing else str(SRC_ROOT)
    )
    env["GPU_MODE"] = "cpu"

    command = [sys.executable, "-m", "models_lstm.cli", *list(arguments)]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _parse_json_lines(output: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        messages.append(json.loads(line))
    return messages


def test_train_score_cli_e2e_logs_and_determinism(tmp_path: Path) -> None:
    normal_csv = tmp_path / "normal.csv"
    anom_csv = tmp_path / "anom.csv"
    _write_contract_csv(normal_csv)
    _write_contract_csv(anom_csv)

    normal_sha = _compute_sha256(normal_csv)
    anom_sha = _compute_sha256(anom_csv)

    run1_dir = tmp_path / "runs" / "exp1"
    run2_dir = tmp_path / "runs" / "exp2"

    train_args = [
        "train",
        "--normal",
        str(normal_csv),
        "--val",
        str(normal_csv),
        "--seed",
        "123",
        "--epochs",
        "1",
    ]

    result_run1 = _run_cli_command([*train_args, "--out", str(run1_dir)])
    assert result_run1.returncode == 0, result_run1.stderr
    train_logs = _parse_json_lines(result_run1.stdout)
    events = {entry.get("event"): entry for entry in train_logs}
    assert "train_start" in events
    assert events["train_start"].get("status") == "started"
    train_start = events["train_start"]
    assert train_start.get("seed") == 123
    assert train_start.get("normal") == str(normal_csv)
    assert train_start.get("val") == str(normal_csv)
    assert train_start.get("output_dir") == str(run1_dir)
    assert train_start.get("normal_sha256") == normal_sha
    assert train_start.get("val_sha256") == normal_sha
    assert train_start.get("input_sha256") == {
        "normal": normal_sha,
        "val": normal_sha,
    }
    assert "train_complete" in events
    train_complete = events["train_complete"]
    assert train_complete.get("status") == "succeeded"
    assert train_complete.get("metrics_path") == str(run1_dir / "metrics.json")
    assert isinstance(train_complete.get("best_epoch"), int)
    for key in [
        "seed",
        "normal",
        "val",
        "output_dir",
        "normal_sha256",
        "val_sha256",
        "input_sha256",
    ]:
        assert train_complete.get(key) == train_start.get(key)

    result_run2 = _run_cli_command([*train_args, "--out", str(run2_dir)])
    assert result_run2.returncode == 0, result_run2.stderr

    scored1 = tmp_path / "scored_run1.csv"
    score1_args = [
        "score",
        "--model",
        str(run1_dir / "best.ckpt"),
        "--in",
        str(anom_csv),
        "--out",
        str(scored1),
    ]
    result_score1 = _run_cli_command(score1_args)
    assert result_score1.returncode == 0, result_score1.stderr
    score_logs = _parse_json_lines(result_score1.stdout)
    score_events = {entry.get("event"): entry for entry in score_logs}
    assert "score_start" in score_events
    score_start = score_events["score_start"]
    assert score_start.get("status") == "started"
    assert score_start.get("model") == str(run1_dir / "best.ckpt")
    assert score_start.get("input") == str(anom_csv)
    assert score_start.get("output") == str(scored1)
    assert score_start.get("seed") == 123
    model_sha = _compute_sha256(run1_dir / "best.ckpt")
    assert score_start.get("model_sha256") == model_sha
    assert score_start.get("input_sha256") == anom_sha
    assert "score_complete" in score_events
    score_complete = score_events["score_complete"]
    assert score_complete.get("status") == "succeeded"
    scored_rows = scored1.read_text(encoding="utf-8").strip().splitlines()
    assert score_complete.get("rows") == len(scored_rows) - 1
    for key in [
        "model",
        "input",
        "output",
        "seed",
        "model_sha256",
        "input_sha256",
    ]:
        assert score_complete.get(key) == score_start.get(key)

    scored2 = tmp_path / "scored_run2.csv"
    score2_args = [
        "score",
        "--model",
        str(run2_dir / "best.ckpt"),
        "--in",
        str(anom_csv),
        "--out",
        str(scored2),
    ]
    result_score2 = _run_cli_command(score2_args)
    assert result_score2.returncode == 0, result_score2.stderr

    assert scored1.exists()
    assert scored2.exists()
    assert scored1.read_bytes() == scored2.read_bytes()


def test_score_cli_failure_nonzero_exit_and_error_log(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    _write_contract_csv(data_csv)

    output_csv = tmp_path / "scored.csv"
    missing_model = tmp_path / "missing.ckpt"
    data_sha = _compute_sha256(data_csv)

    result = _run_cli_command(
        [
            "score",
            "--model",
            str(missing_model),
            "--in",
            str(data_csv),
            "--out",
            str(output_csv),
        ]
    )

    assert result.returncode != 0

    logs = _parse_json_lines(result.stdout)
    events = {entry.get("event"): entry for entry in logs}
    assert "score_start" in events
    score_start = events["score_start"]
    assert score_start.get("status") == "started"
    assert score_start.get("model") == str(missing_model)
    assert score_start.get("input") == str(data_csv)
    assert score_start.get("output") == str(output_csv)
    assert score_start.get("seed") == 0
    assert score_start.get("input_sha256") == data_sha
    assert score_start.get("model_sha256") is None
    assert "score_error" in events
    error_entry = events["score_error"]
    assert error_entry.get("status") == "failed"
    assert error_entry.get("error_code") is not None
    assert "message" in error_entry
    for key in [
        "model",
        "input",
        "output",
        "input_sha256",
        "model_sha256",
        "seed",
    ]:
        assert error_entry.get(key) == score_start.get(key)
