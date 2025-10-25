"""CLI error handling and structured logging tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import math
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_contract import cli
from ds_contract.sessionize import quantile


def _assert_json_lines(stdout: str) -> list[dict[str, object]]:
    lines = [line for line in stdout.strip().splitlines() if line]
    parsed = [json.loads(line) for line in lines]
    for record in parsed:
        assert record["event"] in {"start", "error", "complete"}
        assert "command" in record
    return parsed


def test_validate_missing_input_logs_json_and_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        "\n".join(
            [
                "timestamp_utc: timestamp_utc",
                "uid: uid",
                "session_id: session_id",
                "method: method",
                "path: path",
                "referer: referer",
                "user_agent: user_agent",
                "op_category: op_category",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--seed",
            "7",
            "validate",
            str(tmp_path / "missing.csv"),
            "--map",
            str(mapping_path),
            "--out",
            str(tmp_path / "contract.csv"),
        ]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    records = _assert_json_lines(out)
    assert records[-1]["event"] == "error"
    assert records[-1]["code"] == "INPUT_NOT_FOUND"


def test_sessionize_missing_contract_logs_json_and_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        [
            "--seed",
            "11",
            "sessionize",
            str(tmp_path / "contract.csv"),
            "--out",
            str(tmp_path / "sessioned.csv"),
            "--meta",
            str(tmp_path / "meta.json"),
        ]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    records = _assert_json_lines(out)
    assert records[-1]["event"] == "error"
    assert records[-1]["code"] == "CONTRACT_NOT_FOUND"


def test_deltify_missing_session_logs_json_and_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        [
            "--seed",
            "19",
            "deltify",
            str(tmp_path / "sessioned.csv"),
            "--out",
            str(tmp_path / "deltified.csv"),
            "--meta",
            str(tmp_path / "meta.json"),
        ]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    records = _assert_json_lines(out)
    assert records[-1]["event"] == "error"
    assert records[-1]["code"] == "SESSIONED_NOT_FOUND"


def test_quantile_matches_linear_interpolation() -> None:
    values = [0.1, 2.5, 3.7, 5.9, 7.2]
    q = 0.37
    expected = _linear_interpolate(values, q)
    result = quantile(values, q)
    assert result == pytest.approx(expected, rel=0, abs=1e-12)

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return

    try:
        numpy_expected = float(np.quantile(values, q, method="linear"))
    except TypeError:
        numpy_expected = float(np.quantile(values, q, interpolation="linear"))
    assert result == pytest.approx(numpy_expected, rel=0, abs=1e-12)


def _linear_interpolate(values: list[float], q: float) -> float:
    sorted_values = sorted(values)
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    weight = position - lower_index
    return lower_value * (1 - weight) + upper_value * weight
