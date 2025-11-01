"""Smoke tests for the models-lstm training CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("torch")


def _write_minimal_contract_csv(path: Path) -> None:
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
    text_lines = [",".join(columns)]
    text_lines.extend(",".join(field for field in row) for row in rows)
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def _invoke_train_cli(
    normal_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    epochs: int = 1,
    expect_success: bool = True,
) -> Tuple[str, Path | None, Path | None, subprocess.CompletedProcess[str]]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC_ROOT}{os.pathsep}{existing}" if existing else str(SRC_ROOT)
    )
    env["GPU_MODE"] = "cpu"

    command = [
        sys.executable,
        "-m",
        "models_lstm.cli",
        "train",
        "--normal",
        str(normal_path),
        "--val",
        str(normal_path),
        "--out",
        str(output_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
    ]

    result: subprocess.CompletedProcess[str] = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    checkpoint_path = output_dir / "best.ckpt"
    metrics_path = output_dir / "metrics.json"

    if expect_success:
        if result.returncode != 0:  # pragma: no cover - aid debugging when failing
            raise AssertionError(
                "train command failed",
                result.returncode,
                result.stdout,
                result.stderr,
            )
        assert checkpoint_path.exists(), result.stdout
        assert metrics_path.exists(), result.stdout
        return result.stdout, checkpoint_path, metrics_path, result

    assert result.returncode != 0, result.stdout
    return result.stdout, None, None, result


def _write_short_contract_csv(path: Path) -> None:
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
        ]
    ]
    text_lines = [",".join(columns)]
    text_lines.extend(",".join(field for field in row) for row in rows)
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def _write_contract_csv_with_swapped_header(path: Path) -> None:
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
    swapped = list(columns)
    swapped[0], swapped[1] = swapped[1], swapped[0]
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
        ]
    ]
    text_lines = [",".join(swapped)]
    text_lines.extend(",".join(field for field in row) for row in rows)
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def _write_non_utc_contract_csv(path: Path) -> None:
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
            "2023-01-01T00:00:00+09:00",
            "u1",
            "s1",
            "GET",
            "/a",
            "-",
            "ua",
            "1.1.1.1",
            "cat_a",
        ]
    ]
    text_lines = [",".join(columns)]
    text_lines.extend(",".join(field for field in row) for row in rows)
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def _load_logs(stdout: str) -> Tuple[dict[str, object], ...]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    return tuple(json.loads(line) for line in lines)


def test_train_cli_produces_checkpoint_and_metrics(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.csv"
    _write_minimal_contract_csv(normal_path)

    output_dir = tmp_path / "runs" / "exp1"
    stdout, checkpoint_path, metrics_path, _ = _invoke_train_cli(
        normal_path, output_dir
    )
    assert checkpoint_path is not None
    assert metrics_path is not None

    logs = _load_logs(stdout)
    assert len(logs) == 2
    start_log, end_log = logs
    assert start_log["event"] == "train_start"
    assert start_log["status"] == "started"
    assert start_log["seed"] == 42
    assert start_log["output_dir"] == str(output_dir)
    assert start_log["normal_sha256"] == start_log["val_sha256"]
    assert "input_sha256" in start_log

    assert end_log["event"] == "train_complete"
    assert end_log["status"] == "succeeded"
    assert end_log["metrics_path"] == str(metrics_path)
    assert end_log["normal_sha256"] == start_log["normal_sha256"]
    assert end_log["val_sha256"] == start_log["val_sha256"]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "best_epoch" in metrics
    assert "loss" in metrics
    assert "f1" in metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "detection_delay" in metrics
    detection_delay = metrics["detection_delay"]
    assert isinstance(detection_delay, (int, float))
    assert detection_delay >= 0.0
    thresholds = metrics.get("thresholds")
    assert isinstance(thresholds, dict)
    assert thresholds.get("strategy") == "quantile"
    assert thresholds.get("method") == "linear"
    assert thresholds.get("alpha") == pytest.approx(0.05)
    per_category = thresholds.get("per_category")
    assert isinstance(per_category, dict) and per_category
    for category, values in per_category.items():
        assert isinstance(category, str) and category
        assert isinstance(values, dict)
        assert "tau_lo" in values
        assert "tau_hi" in values
    if metrics["pr_auc"] is not None:
        assert 0.0 <= metrics["pr_auc"] <= 1.0
    if metrics["roc_auc"] is not None:
        assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_train_cli_is_deterministic_given_same_seed(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.csv"
    _write_minimal_contract_csv(normal_path)

    output_dir_one = tmp_path / "runs" / "exp1"
    _, checkpoint_one, metrics_one, _ = _invoke_train_cli(normal_path, output_dir_one)
    assert checkpoint_one is not None
    assert metrics_one is not None

    output_dir_two = tmp_path / "runs" / "exp2"
    _, checkpoint_two, metrics_two, _ = _invoke_train_cli(normal_path, output_dir_two)
    assert checkpoint_two is not None
    assert metrics_two is not None

    assert checkpoint_one.read_bytes() == checkpoint_two.read_bytes()
    assert json.loads(metrics_one.read_text(encoding="utf-8")) == json.loads(
        metrics_two.read_text(encoding="utf-8")
    )


def test_train_cli_logs_error_payload_on_failure(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.csv"
    _write_short_contract_csv(normal_path)

    output_dir = tmp_path / "runs" / "exp1"
    stdout, _, _, result = _invoke_train_cli(
        normal_path, output_dir, expect_success=False
    )

    logs = _load_logs(stdout)
    assert len(logs) == 2
    start_log, error_log = logs
    assert start_log["event"] == "train_start"
    assert start_log["status"] == "started"
    assert error_log["event"] == "train_error"
    assert error_log["status"] == "failed"
    assert error_log["error_code"] == "ValueError"
    assert "message" in error_log
    assert error_log["seed"] == 42
    assert error_log["normal_sha256"] == start_log["normal_sha256"]
    assert error_log["val_sha256"] == start_log["val_sha256"]
    assert result.returncode != 0


def test_train_cli_rejects_contract_with_wrong_header_order(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.csv"
    _write_contract_csv_with_swapped_header(normal_path)

    output_dir = tmp_path / "runs" / "exp1"
    stdout, _, _, result = _invoke_train_cli(
        normal_path, output_dir, expect_success=False
    )

    logs = _load_logs(stdout)
    assert len(logs) == 2
    _, error_log = logs
    assert error_log["event"] == "train_error"
    assert error_log["status"] == "failed"
    assert error_log["error_code"] == "ValueError"
    message = str(error_log.get("message"))
    assert "header order must exactly match contract specification" in message
    assert "expected header order" in message
    assert result.returncode != 0


def test_train_cli_rejects_non_utc_timestamp(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.csv"
    _write_non_utc_contract_csv(normal_path)

    output_dir = tmp_path / "runs" / "exp1"
    stdout, _, _, result = _invoke_train_cli(
        normal_path, output_dir, expect_success=False
    )

    logs = _load_logs(stdout)
    assert len(logs) == 2
    _, error_log = logs
    assert error_log["event"] == "train_error"
    assert error_log["status"] == "failed"
    assert error_log["error_code"] == "ValueError"
    message = str(error_log.get("message"))
    assert "Please re-export UTC-normalized contract data." in message
    assert "non-UTC offset" in message or "timezone" in message
    assert result.returncode != 0
