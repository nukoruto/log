import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_contract.dt import DELTIFIED_COLUMNS, DeltifyResult, deltify_session_rows
from ds_contract.sessionize import EPSILON, SESSION_COLUMNS
from ds_contract.cli import main


def _build_session_rows(
    uid: str,
    start: datetime,
    deltas: list[float],
    method: str = "GET",
    path: str = "/index",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_time = start
    for index, delta in enumerate(deltas):
        if index == 0:
            current_time = start
        else:
            current_time = current_time + timedelta(seconds=delta)
        row = {
            "timestamp_utc": current_time.isoformat(),
            "uid": uid,
            "session_id": f"{uid}-0001",
            "method": method,
            "path": path,
            "referer": "-",
            "user_agent": "ua",
            "ip": "203.0.113.1",
            "op_category": "view",
            "delta_t_seconds": f"{delta:.6f}",
            "log_delta_t": f"{math.log(delta + EPSILON):.12f}",
        }
        rows.append(row)
    return rows


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _mad(values: list[float], median: float) -> float:
    deviations = [abs(value - median) for value in values]
    if not deviations:
        return 0.0
    return _median(deviations)


def test_deltify_produces_expected_z_scores() -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    user_a = _build_session_rows(
        "user-a",
        base_time,
        [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    user_b = _build_session_rows(
        "user-b",
        base_time + timedelta(minutes=10),
        [0.0, 0.5, 0.5, 0.5],
    )
    rows = user_a + user_b

    result = deltify_session_rows(rows)
    assert isinstance(result, DeltifyResult)
    assert len(result.rows) == len(rows)

    log_values = [math.log(float(row["delta_t_seconds"]) + EPSILON) for row in rows]
    global_median = _median(log_values)
    global_mad = _mad(log_values, global_median) or 1e-9

    per_user = {
        "user-a": [math.log(float(row["delta_t_seconds"]) + EPSILON) for row in user_a],
        "user-b": [math.log(float(row["delta_t_seconds"]) + EPSILON) for row in user_b],
    }

    for row in result.rows:
        delta = float(row["delta_t_seconds"])
        log_delta = math.log(delta + EPSILON)
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            assert float(row["robust_z"]) == pytest.approx(0.0, abs=1e-6)
            assert float(row["robust_z_clipped"]) == pytest.approx(0.0, abs=1e-6)
            continue

        values = per_user[row["uid"]]
        if len(values) >= 5:
            median = _median(values)
            mad = _mad(values, median) or global_mad
        else:
            median = global_median
            mad = global_mad

        expected = (log_delta - median) / (1.4826 * mad)
        assert float(row["robust_z"]) == pytest.approx(expected, rel=1e-6, abs=1e-6)

    assert "user-b" in result.meta["fallback_users"]
    assert result.meta["group"] == "uid"


def test_deltify_clips_extreme_values() -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = _build_session_rows(
        "clipper",
        base_time,
        [0.0, 0.01, 0.01, 0.01, 0.01, 5000.0],
    )

    result = deltify_session_rows(rows)

    extreme = [row for row in result.rows if float(row["delta_t_seconds"]) == 5000.0][0]
    assert float(extreme["robust_z"]) > 5.0
    assert float(extreme["robust_z_clipped"]) == pytest.approx(5.0, abs=1e-6)


def test_cli_deltify_writes_csv_and_meta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = _build_session_rows(
        "cli-user",
        base_time,
        [0.0, 1.0, 2.0, 3.0, 5.0],
    )

    input_path = tmp_path / "sessioned.csv"
    output_path = tmp_path / "deltified.csv"
    meta_path = tmp_path / "meta_dt.json"

    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SESSION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    exit_code = main(
        [
            "--seed",
            "42",
            "deltify",
            str(input_path),
            "--out",
            str(output_path),
            "--meta",
            str(meta_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert meta_path.exists()

    with output_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == DELTIFIED_COLUMNS
        rows_out = list(reader)
    assert len(rows_out) == len(rows)
    assert all("robust_z" in row for row in rows_out)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["clip_range"] == [-5.0, 5.0]
    assert meta["seed"] == 42
    assert "input_sha256" in meta

    assert "start" in captured.out
    assert "complete" in captured.out
