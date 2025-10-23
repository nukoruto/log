from __future__ import annotations

import json
from pathlib import Path

import matlab_bridge.cli as cli

from .test_export_mat import HEADER, make_row, write_csv


def _run_cli(args: list[str], capsys) -> tuple[int, list[dict[str, object]]]:
    exit_code = cli.main(args)
    captured = capsys.readouterr()
    stdout = captured.out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    return exit_code, logs


def test_cli_rejects_nan_values(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "scored.csv"
    out_path = tmp_path / "ref.mat"
    meta_path = tmp_path / "meta.json"

    write_csv(
        csv_path,
        [
            ",".join(HEADER),
            make_row(
                "2024-01-01T00:00:00Z",
                "u0",
                "s0",
                "catA",
                "0.0",
                "0.1",
                "0.05",
                "0.20",
                "NaN",
                "0",
                "0",
            ),
        ],
    )

    exit_code, logs = _run_cli(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
            "--seed",
            "11",
        ],
        capsys,
    )

    assert exit_code == 1
    assert not out_path.exists()
    assert not meta_path.exists()
    assert logs[0]["event"] == "export.start"
    assert logs[-1]["event"] == "export.error"
    assert logs[-1]["error"] == "validation_error"
    message = str(logs[-1].get("message", ""))
    assert "NaN" in message


def test_cli_rejects_missing_signal_value(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "scored.csv"
    out_path = tmp_path / "ref.mat"
    meta_path = tmp_path / "meta.json"

    write_csv(
        csv_path,
        [
            ",".join(HEADER),
            make_row(
                "2024-01-01T00:00:00Z",
                "u1",
                "s1",
                "catB",
                "0.0",
                "0.1",
                "0.05",
                "0.20",
                "0.5",
                "1",
                "",
            ),
        ],
    )

    exit_code, logs = _run_cli(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
            "--seed",
            "12",
        ],
        capsys,
    )

    assert exit_code == 1
    assert not out_path.exists()
    assert not meta_path.exists()
    assert logs[0]["event"] == "export.start"
    assert logs[-1]["event"] == "export.error"
    assert logs[-1]["error"] == "validation_error"
    message = str(logs[-1].get("message", ""))
    assert "flag_dt" in message


def test_cli_rejects_non_monotonic_time(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "scored.csv"
    out_path = tmp_path / "ref.mat"
    meta_path = tmp_path / "meta.json"

    write_csv(
        csv_path,
        [
            ",".join(HEADER),
            make_row(
                "2024-01-01T00:00:01Z",
                "u2",
                "s2",
                "catC",
                "0.0",
                "0.1",
                "0.05",
                "0.20",
                "0.9",
                "0",
                "0",
            ),
            make_row(
                "2024-01-01T00:00:00Z",
                "u2",
                "s2",
                "catC",
                "0.0",
                "0.1",
                "0.05",
                "0.20",
                "1.0",
                "0",
                "0",
            ),
        ],
    )

    exit_code, logs = _run_cli(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
            "--seed",
            "13",
        ],
        capsys,
    )

    assert exit_code == 1
    assert not out_path.exists()
    assert not meta_path.exists()
    assert logs[0]["event"] == "export.start"
    assert logs[-1]["event"] == "export.error"
    assert logs[-1]["error"] == "validation_error"
    message = str(logs[-1].get("message", ""))
    assert "strictly increasing" in message
