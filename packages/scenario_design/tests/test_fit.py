import csv
import json
import math
import pickle
import sys
from pathlib import Path

import pytest

# Ensure the scenario_design package is importable without installation.
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scenario_design import fit  # noqa: E402
from scenario_design import cli  # noqa: E402


def write_deltified_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_estimate_statistics(tmp_path: Path) -> None:
    csv_path = tmp_path / "deltified.csv"
    rows = [
        {
            "timestamp_utc": "2024-01-01T00:00:00Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "POST",
            "path": "/auth",
            "referer": "-",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "AUTH",
            "delta_t": "0.1",
        },
        {
            "timestamp_utc": "2024-01-01T00:00:01Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "GET",
            "path": "/read",
            "referer": "/auth",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "READ",
            "delta_t": "0.5",
        },
        {
            "timestamp_utc": "2024-01-01T00:00:03Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "GET",
            "path": "/read",
            "referer": "/read",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "READ",
            "delta_t": "1.0",
        },
        {
            "timestamp_utc": "2024-01-01T00:00:06Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "POST",
            "path": "/update",
            "referer": "/read",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "UPDATE",
            "delta_t": "0.2",
        },
        {
            "timestamp_utc": "2024-01-02T00:00:00Z",
            "uid": "u2",
            "session_id": "s2",
            "method": "POST",
            "path": "/auth",
            "referer": "-",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "AUTH",
            "delta_t": "0.2",
        },
        {
            "timestamp_utc": "2024-01-02T00:00:01Z",
            "uid": "u2",
            "session_id": "s2",
            "method": "GET",
            "path": "/read",
            "referer": "/auth",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "READ",
            "delta_t": "0.4",
        },
    ]
    write_deltified_csv(csv_path, rows)

    stats = fit.estimate_statistics(csv_path)
    stats_dict = stats.to_dict()

    assert stats_dict["ops"] == ["AUTH", "READ", "UPDATE"]
    assert math.isclose(stats_dict["pi"]["AUTH"], 7 / 9, rel_tol=1e-5)
    assert math.isclose(stats_dict["pi"]["READ"], 1 / 9, rel_tol=1e-5)
    assert math.isclose(stats_dict["pi"]["UPDATE"], 1 / 9, rel_tol=1e-5)

    assert math.isclose(stats_dict["A"]["AUTH"]["READ"], 7 / 9, rel_tol=1e-5)
    assert math.isclose(stats_dict["A"]["AUTH"]["AUTH"], 1 / 9, rel_tol=1e-5)
    assert math.isclose(stats_dict["A"]["READ"]["READ"], 4 / 9, rel_tol=1e-5)
    assert math.isclose(stats_dict["A"]["READ"]["UPDATE"], 4 / 9, rel_tol=1e-5)
    assert math.isclose(stats_dict["A"]["UPDATE"]["AUTH"], 1 / 3, rel_tol=1e-5)

    eps = fit.LOG_EPS
    values_by_cat = {
        "AUTH": [0.1, 0.2],
        "READ": [0.5, 1.0, 0.4],
        "UPDATE": [0.2],
    }
    for cat, values in values_by_cat.items():
        logs = [math.log(value + eps) for value in values]
        expected_mu = sum(logs) / len(logs)
        variance = sum((value - expected_mu) ** 2 for value in logs) / len(logs)
        expected_sigma = math.sqrt(variance)
        assert math.isclose(
            stats_dict["dt"]["mu"][cat], expected_mu, rel_tol=1e-9, abs_tol=1e-9
        )
        assert math.isclose(
            stats_dict["dt"]["sigma"][cat], expected_sigma, rel_tol=1e-9, abs_tol=1e-9
        )

    assert "input_sha256" in stats_dict
    assert stats_dict["algo_version"].startswith("B1")


def test_estimate_statistics_unknown_op(tmp_path: Path) -> None:
    csv_path = tmp_path / "deltified.csv"
    rows = [
        {
            "timestamp_utc": "2024-01-01T00:00:00Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "POST",
            "path": "/auth",
            "referer": "-",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "UNKNOWN",
            "delta_t": "0.1",
        }
    ]
    write_deltified_csv(csv_path, rows)

    with pytest.raises(ValueError):
        fit.estimate_statistics(csv_path)


def test_cli_fit_produces_stats_and_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "deltified.csv"
    rows = [
        {
            "timestamp_utc": "2024-01-01T00:00:00Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "POST",
            "path": "/auth",
            "referer": "-",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "AUTH",
            "delta_t": "0.1",
        },
        {
            "timestamp_utc": "2024-01-01T00:00:01Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "GET",
            "path": "/read",
            "referer": "/auth",
            "user_agent": "UA",
            "ip": "127.0.0.1",
            "op_category": "READ",
            "delta_t": "0.4",
        },
    ]
    write_deltified_csv(csv_path, rows)

    out_path = tmp_path / "stats.pkl"
    cli.main(["fit", str(csv_path), "--out", str(out_path)])

    captured = capsys.readouterr()
    output_lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(output_lines) == 2
    for line in output_lines:
        payload = json.loads(line)
        assert payload["event"] in {"start", "complete"}

    assert out_path.exists()
    with out_path.open("rb") as f:
        stats = pickle.load(f)
    assert stats["pi"]
    assert stats["A"]
    assert stats["dt"]
