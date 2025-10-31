"""CLI error handling and structured logging tests."""

from __future__ import annotations

import json
import math
import sys
import random
from io import StringIO
from pathlib import Path
from typing import Any, Callable, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_contract.contract import validate_cli
from ds_contract.dt import deltify_cli
from ds_contract.sessionize import quantile, sessionize_cli


def _assert_json_lines(stdout: str) -> list[dict[str, Any]]:
    lines = [line for line in stdout.strip().splitlines() if line]
    parsed: list[dict[str, Any]] = [json.loads(line) for line in lines]
    for record in parsed:
        assert record["event"] in {"start", "error", "complete"}
        assert "command" in record
    return parsed


def _run_and_assert_seed(seed: int, runner: Callable[[StringIO], int]) -> int:
    random_state = random.getstate()
    np_module: Any | None = None
    np_state: Any | None = None
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - numpy optional
        np_module = None
    else:
        np_module = _np
        np_state = _np.random.get_state()

    try:
        random.seed(seed + 12345)
        if np_module is not None:
            np_module.random.seed(seed + 54321)

        buffer = StringIO()
        exit_code = runner(buffer)
        records = _assert_json_lines(buffer.getvalue())
        assert records[0]["seed"] == seed

        expected_random = random.Random(seed).random()
        actual_random = random.random()
        assert actual_random == pytest.approx(expected_random, rel=0, abs=1e-12)

        if np_module is not None:
            expected_np = float(np_module.random.RandomState(seed).random_sample())
            actual_np = float(np_module.random.random_sample())
            assert actual_np == pytest.approx(expected_np, rel=0, abs=1e-12)

        return exit_code
    finally:
        random.setstate(random_state)
        if np_module is not None and np_state is not None:
            np_module.random.set_state(np_state)


def test_validate_missing_input_logs_json_and_nonzero_exit(tmp_path: Path) -> None:
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

    buffer = StringIO()
    exit_code = validate_cli(
        str(tmp_path / "missing.csv"),
        mapping=str(mapping_path),
        output=str(tmp_path / "contract.csv"),
        meta=str(tmp_path / "meta.json"),
        seed=7,
        stream=buffer,
    )

    assert exit_code == 1
    records = _assert_json_lines(buffer.getvalue())
    assert records[0]["event"] == "start"
    start_details_obj = records[0]["details"]
    assert isinstance(start_details_obj, dict)
    start_details = cast(dict[str, Any], start_details_obj)
    assert start_details["input"] == str(tmp_path / "missing.csv")
    assert records[-1]["event"] == "error"
    assert records[-1]["code"] == "INPUT_NOT_FOUND"
    assert records[-1]["command"] == "validate"


def test_sessionize_missing_contract_logs_json_and_nonzero_exit(tmp_path: Path) -> None:
    buffer = StringIO()
    exit_code = sessionize_cli(
        str(tmp_path / "contract.csv"),
        output=str(tmp_path / "sessioned.csv"),
        meta=str(tmp_path / "meta.json"),
        seed=11,
        stream=buffer,
    )

    assert exit_code == 1
    records = _assert_json_lines(buffer.getvalue())
    assert records[0]["event"] == "start"
    start_details_obj = records[0]["details"]
    assert isinstance(start_details_obj, dict)
    start_details = cast(dict[str, Any], start_details_obj)
    assert start_details["input"] == str(tmp_path / "contract.csv")
    assert records[-1]["event"] == "error"
    assert records[-1]["code"] == "CONTRACT_NOT_FOUND"
    assert records[-1]["command"] == "sessionize"


def test_deltify_missing_session_logs_json_and_nonzero_exit(tmp_path: Path) -> None:
    buffer = StringIO()
    exit_code = deltify_cli(
        str(tmp_path / "sessioned.csv"),
        output=str(tmp_path / "deltified.csv"),
        meta=str(tmp_path / "meta.json"),
        seed=19,
        stream=buffer,
    )

    assert exit_code == 1
    records = _assert_json_lines(buffer.getvalue())
    assert records[0]["event"] == "start"
    start_details_obj = records[0]["details"]
    assert isinstance(start_details_obj, dict)
    start_details = cast(dict[str, Any], start_details_obj)
    assert start_details["input"] == str(tmp_path / "sessioned.csv")
    assert records[-1]["event"] == "error"
    assert records[-1]["code"] == "SESSIONED_NOT_FOUND"
    assert records[-1]["command"] == "deltify"


def test_validate_missing_required_argument_logs_argument_error(tmp_path: Path) -> None:
    buffer = StringIO()
    exit_code = validate_cli(
        str(tmp_path / "input.csv"),
        mapping=None,
        output=str(tmp_path / "contract.csv"),
        meta=None,
        seed=3,
        stream=buffer,
    )

    assert exit_code == 2
    records = _assert_json_lines(buffer.getvalue())
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "error"
    assert record["code"] == "ARGUMENT_ERROR"
    assert record["command"] == "validate"
    assert record["seed"] == 3
    assert "--map" in str(record["message"])
    hint = record.get("hint")
    assert isinstance(hint, str)
    assert "help" in hint.lower()


def test_validate_cli_sets_global_seed(tmp_path: Path) -> None:
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

    exit_code = _run_and_assert_seed(
        101,
        lambda stream: validate_cli(
            str(tmp_path / "missing.csv"),
            mapping=str(mapping_path),
            output=str(tmp_path / "contract.csv"),
            meta=str(tmp_path / "meta.json"),
            seed=101,
            stream=stream,
        ),
    )

    assert exit_code == 1


def test_sessionize_missing_required_argument_logs_argument_error(
    tmp_path: Path,
) -> None:
    buffer = StringIO()
    exit_code = sessionize_cli(
        str(tmp_path / "contract.csv"),
        output=str(tmp_path / "sessioned.csv"),
        meta=None,
        seed=13,
        stream=buffer,
    )

    assert exit_code == 2
    records = _assert_json_lines(buffer.getvalue())
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "error"
    assert record["code"] == "ARGUMENT_ERROR"
    assert record["command"] == "sessionize"
    assert record["seed"] == 13
    assert "--meta" in str(record["message"])
    hint = record.get("hint")
    assert isinstance(hint, str)
    assert "help" in hint.lower()


def test_sessionize_cli_sets_global_seed(tmp_path: Path) -> None:
    exit_code = _run_and_assert_seed(
        211,
        lambda stream: sessionize_cli(
            str(tmp_path / "contract.csv"),
            output=str(tmp_path / "sessioned.csv"),
            meta=str(tmp_path / "meta.json"),
            seed=211,
            stream=stream,
        ),
    )

    assert exit_code == 1


def test_deltify_missing_required_argument_logs_argument_error(tmp_path: Path) -> None:
    buffer = StringIO()
    exit_code = deltify_cli(
        str(tmp_path / "sessioned.csv"),
        output=str(tmp_path / "deltified.csv"),
        meta=None,
        seed=21,
        stream=buffer,
    )

    assert exit_code == 2
    records = _assert_json_lines(buffer.getvalue())
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "error"
    assert record["code"] == "ARGUMENT_ERROR"
    assert record["command"] == "deltify"
    assert record["seed"] == 21
    assert "--meta" in str(record["message"])
    hint = record.get("hint")
    assert isinstance(hint, str)
    assert "help" in hint.lower()


def test_deltify_cli_sets_global_seed(tmp_path: Path) -> None:
    exit_code = _run_and_assert_seed(
        307,
        lambda stream: deltify_cli(
            str(tmp_path / "sessioned.csv"),
            output=str(tmp_path / "deltified.csv"),
            meta=str(tmp_path / "meta.json"),
            seed=307,
            stream=stream,
        ),
    )

    assert exit_code == 1


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
