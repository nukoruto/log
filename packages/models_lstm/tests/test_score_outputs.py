"""Tests for the score command output generation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

import pytest


torch = pytest.importorskip("torch")


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from models_lstm.score import run_score_command  # noqa: E402
from models_lstm.train import AnomalyDetectorModel  # noqa: E402


def _write_contract_csv(path: Path, rows: List[List[str]]) -> None:
    header = [
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
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _create_zero_checkpoint(
    model_path: Path, vocab: List[str], *, seed: int | None = None
) -> None:
    embed_dim = 4
    hidden_dim = 4
    num_layers = 1
    config = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": 0.0,
        "clip_value": 5.0,
    }
    if seed is not None:
        config["seed"] = seed
    model = AnomalyDetectorModel(
        num_categories=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=float(config["dropout"]),
    )
    zero_state = {
        key: torch.zeros_like(param) for key, param in model.state_dict().items()
    }
    payload = {
        "model_state": zero_state,
        "encoder_vocab": list(vocab),
        "config": config,
    }
    torch.save(payload, model_path)


def test_score_outputs_expected_values(tmp_path: Path) -> None:
    rows = [
        [
            "2024-01-01T00:00:00+00:00",
            "user-1",
            "session-1",
            "GET",
            "/login",
            "-",
            "agent",
            "127.0.0.1",
            "login",
        ],
        [
            "2024-01-01T00:00:05+00:00",
            "user-1",
            "session-1",
            "GET",
            "/click",
            "-",
            "agent",
            "127.0.0.1",
            "click",
        ],
        [
            "2024-01-01T00:00:15+00:00",
            "user-1",
            "session-1",
            "POST",
            "/logout",
            "-",
            "agent",
            "127.0.0.1",
            "logout",
        ],
    ]

    contract_path = tmp_path / "anom.csv"
    _write_contract_csv(contract_path, rows)

    model_dir = tmp_path / "runs" / "exp1"
    model_dir.mkdir(parents=True)
    checkpoint_path = model_dir / "best.ckpt"
    metrics_path = model_dir / "metrics.json"

    vocab = ["click", "login", "logout"]
    _create_zero_checkpoint(checkpoint_path, vocab, seed=123)

    metrics = {
        "thresholds": {
            "alpha": 0.05,
            "strategy": "quantile",
            "method": "linear",
            "per_category": {
                "click": {"tau_lo": 0.0, "tau_hi": 0.25},
                "login": {"tau_lo": 0.0, "tau_hi": 0.25},
                "logout": {"tau_lo": 0.0, "tau_hi": 0.5},
            },
        }
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    run_score_command(
        model=checkpoint_path,
        input_path=contract_path,
        output_path=tmp_path / "scored.csv",
    )

    scored_rows = list(
        csv.DictReader((tmp_path / "scored.csv").open("r", encoding="utf-8"))
    )

    assert [
        "timestamp_utc",
        "uid",
        "session_id",
        "op_category",
        "z",
        "z_hat",
        "s_cls",
        "s_time",
        "S",
        "flag_cls",
        "flag_dt",
    ] == list(scored_rows[0].keys())

    expected_prob_gap = 2.0 / 3.0

    first = scored_rows[0]
    assert first["op_category"] == "login"
    assert float(first["z"]) == pytest.approx(-0.67448975, rel=1e-6)
    assert float(first["z_hat"]) == 0.0
    assert float(first["s_cls"]) == 0.0
    assert float(first["s_time"]) == 0.0
    assert float(first["S"]) == 0.0
    assert int(first["flag_cls"]) == 0
    assert int(first["flag_dt"]) == 0

    second = scored_rows[1]
    assert second["op_category"] == "click"
    assert float(second["z"]) == pytest.approx(0.0, abs=1e-9)
    assert float(second["z_hat"]) == pytest.approx(0.0, abs=1e-9)
    assert float(second["s_cls"]) == pytest.approx(expected_prob_gap, rel=1e-6)
    assert float(second["s_time"]) == pytest.approx(0.0, abs=1e-9)
    assert float(second["S"]) == pytest.approx(expected_prob_gap * 0.5, rel=1e-6)
    assert int(second["flag_cls"]) == 0
    assert int(second["flag_dt"]) == 0

    third = scored_rows[2]
    assert third["op_category"] == "logout"
    assert float(third["z"]) == pytest.approx(0.67448975, rel=1e-6)
    assert float(third["z_hat"]) == pytest.approx(0.0, abs=1e-9)
    assert float(third["s_cls"]) == pytest.approx(expected_prob_gap, rel=1e-6)
    assert float(third["s_time"]) == pytest.approx(0.67448975, rel=1e-6)
    assert float(third["S"]) == pytest.approx(
        0.5 * (expected_prob_gap + 0.67448975), rel=1e-6
    )
    assert int(third["flag_cls"]) == 1
    assert int(third["flag_dt"]) == 1

    # Ensure determinism by running the command a second time and comparing content
    second_output = tmp_path / "scored_again.csv"
    run_score_command(
        model=checkpoint_path, input_path=contract_path, output_path=second_output
    )

    assert (tmp_path / "scored.csv").read_text(
        encoding="utf-8"
    ) == second_output.read_text(encoding="utf-8")


def test_run_score_command_logs_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        [
            "2024-01-01T00:00:00+00:00",
            "user-1",
            "session-1",
            "GET",
            "/login",
            "-",
            "agent",
            "127.0.0.1",
            "login",
        ],
        [
            "2024-01-01T00:00:05+00:00",
            "user-1",
            "session-1",
            "GET",
            "/click",
            "-",
            "agent",
            "127.0.0.1",
            "click",
        ],
    ]

    contract_path = tmp_path / "anom.csv"
    _write_contract_csv(contract_path, rows)

    model_dir = tmp_path / "runs" / "exp1"
    model_dir.mkdir(parents=True)
    checkpoint_path = model_dir / "best.ckpt"
    metrics_path = model_dir / "metrics.json"

    vocab = ["click", "login"]
    expected_seed = 321
    _create_zero_checkpoint(checkpoint_path, vocab, seed=expected_seed)
    metrics_path.write_text(json.dumps({"thresholds": {}}), encoding="utf-8")

    run_score_command(
        model=checkpoint_path,
        input_path=contract_path,
        output_path=tmp_path / "scored.csv",
    )

    captured = capsys.readouterr()
    logs = [
        json.loads(line) for line in captured.out.strip().splitlines() if line.strip()
    ]

    assert len(logs) >= 2
    start_log = next(item for item in logs if item["event"] == "score_start")
    complete_log = next(item for item in logs if item["event"] == "score_complete")

    assert start_log["seed"] == expected_seed
    assert complete_log["seed"] == expected_seed
