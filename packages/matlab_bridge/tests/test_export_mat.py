"""Tests for the matlab-bridge export command."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from math import isclose
from pathlib import Path

import pytest

from .mat_stub import loadmat

import matlab_bridge.cli as cli


HEADER = [
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
]


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_row(
    timestamp: str,
    uid: str,
    session_id: str,
    op_category: str,
    z: str,
    z_hat: str,
    s_cls: str,
    s_time: str,
    score: str,
    flag_cls: str,
    flag_dt: str,
) -> str:
    return ",".join(
        [
            timestamp,
            uid,
            session_id,
            op_category,
            z,
            z_hat,
            s_cls,
            s_time,
            score,
            flag_cls,
            flag_dt,
        ]
    )


def test_export_help_marks_required_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["export", "--help"])

    assert excinfo.value.code == 0
    help_output = capsys.readouterr().out
    assert "--meta META" in help_output
    assert "--seed SEED" in help_output
    assert "[--meta META]" not in help_output
    assert "[--seed SEED]" not in help_output


def test_export_creates_mat_file(tmp_path: Path, capsys) -> None:
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
                "1.00",
                "0",
                "0",
            ),
            make_row(
                "2024-01-01T00:00:01Z",
                "u0",
                "s0",
                "catA",
                "0.1",
                "0.2",
                "0.06",
                "0.40",
                "1.50",
                "0",
                "1",
            ),
            make_row(
                "2024-01-01T00:00:02Z",
                "u0",
                "s0",
                "catA",
                "0.2",
                "0.3",
                "0.07",
                "0.60",
                "2.00",
                "1",
                "0",
            ),
        ],
    )

    expected_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    exit_code = cli.main(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
            "--seed",
            "123",
        ]
    )

    assert exit_code == 0
    assert out_path.exists()
    assert meta_path.exists()

    stdout = capsys.readouterr().out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    assert logs[0]["event"] == "export.start"
    assert logs[0]["seed"] == 123
    assert logs[0]["input_sha256"] == expected_sha
    assert logs[0]["meta"] == str(meta_path)
    assert logs[-1]["event"] == "export.complete"
    assert logs[-1]["seed"] == 123
    assert logs[-1]["input_sha256"] == expected_sha
    assert logs[-1]["meta"] == str(meta_path)

    mat = loadmat(out_path)
    for key in {"ref", "y_lstm", "y_pid", "t"}:
        assert key in mat

    ref = [float(x) for x in mat["ref"].flatten()]
    y_lstm = [float(x) for x in mat["y_lstm"].flatten()]
    y_pid = [float(x) for x in mat["y_pid"].flatten()]
    t_values = [float(x) for x in mat["t"].flatten()]

    assert all(
        isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(ref, [1.0, 1.5, 2.0])
    )
    assert all(
        isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(y_lstm, [0.2, 0.4, 0.6])
    )
    assert all(
        isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(y_pid, [0.0, 1.0, 0.0])
    )
    assert all(
        isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(t_values, [0.0, 1.0, 2.0])
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["seed"] == 123
    assert meta["algo_version"] == cli.ALGO_VERSION
    assert meta["input_sha256"] == expected_sha
    assert meta["row_count"] == 3
    assert meta["output_mat"] == out_path.name
    parsed_timestamp = datetime.fromisoformat(
        meta["generated_at"].replace("Z", "+00:00")
    )
    assert parsed_timestamp.tzinfo is not None
    assert meta["generated_at"] == "2024-01-01T00:00:02Z"


def test_export_is_deterministic_for_same_seed(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "scored.csv"
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
                "1.00",
                "0",
                "0",
            ),
            make_row(
                "2024-01-01T00:00:01Z",
                "u0",
                "s0",
                "catA",
                "0.1",
                "0.2",
                "0.06",
                "0.40",
                "1.50",
                "0",
                "1",
            ),
            make_row(
                "2024-01-01T00:00:02Z",
                "u0",
                "s0",
                "catA",
                "0.2",
                "0.3",
                "0.07",
                "0.60",
                "2.00",
                "1",
                "0",
            ),
        ],
    )

    first_dir = tmp_path / "run1"
    second_dir = tmp_path / "run2"
    first_dir.mkdir()
    second_dir.mkdir()

    first_args = [
        "export",
        "--in",
        str(csv_path),
        "--out",
        str(first_dir / "ref.mat"),
        "--meta",
        str(first_dir / "meta.json"),
        "--seed",
        "777",
    ]
    second_args = [
        "export",
        "--in",
        str(csv_path),
        "--out",
        str(second_dir / "ref.mat"),
        "--meta",
        str(second_dir / "meta.json"),
        "--seed",
        "777",
    ]

    exit_code_first = cli.main(first_args)
    assert exit_code_first == 0
    capsys.readouterr()

    exit_code_second = cli.main(second_args)
    assert exit_code_second == 0
    capsys.readouterr()

    first_bytes = (first_dir / "ref.mat").read_bytes()
    second_bytes = (second_dir / "ref.mat").read_bytes()
    assert first_bytes == second_bytes

    first_meta = json.loads((first_dir / "meta.json").read_text(encoding="utf-8"))
    second_meta = json.loads((second_dir / "meta.json").read_text(encoding="utf-8"))
    assert first_meta == second_meta


def test_export_resamples_to_median_grid(tmp_path: Path, capsys) -> None:
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
                "0.0",
                "0.0",
                "0.50",
                "0.0",
                "0",
                "0",
            ),
            make_row(
                "2024-01-01T00:00:01Z",
                "u0",
                "s0",
                "catA",
                "0.1",
                "0.1",
                "0.1",
                "1.50",
                "1.0",
                "0",
                "0",
            ),
            make_row(
                "2024-01-01T00:00:01.900000Z",
                "u0",
                "s0",
                "catA",
                "0.2",
                "0.2",
                "0.2",
                "2.40",
                "1.9",
                "1",
                "1",
            ),
            make_row(
                "2024-01-01T00:00:03.100000Z",
                "u0",
                "s0",
                "catA",
                "0.3",
                "0.3",
                "0.3",
                "3.60",
                "3.1",
                "1",
                "1",
            ),
        ],
    )

    exit_code = cli.main(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
            "--seed",
            "321",
        ]
    )

    assert exit_code == 0

    mat = loadmat(out_path)
    t_values = [float(x) for x in mat["t"].flatten()]
    ref_values = [float(x) for x in mat["ref"].flatten()]
    lstm_values = [float(x) for x in mat["y_lstm"].flatten()]
    pid_values = [float(x) for x in mat["y_pid"].flatten()]

    expected_time = [0.0, 1.0, 2.0, 3.0]
    expected_ref = [0.0, 1.0, 2.0, 3.0]
    expected_lstm = [0.5, 1.5, 2.5, 3.5]
    expected_pid = [0.0, 0.0, 1.0, 1.0]

    assert all(
        isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        for actual, expected in zip(t_values, expected_time)
    )
    assert all(
        isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        for actual, expected in zip(ref_values, expected_ref)
    )
    assert all(
        isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        for actual, expected in zip(lstm_values, expected_lstm)
    )
    assert all(
        isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        for actual, expected in zip(pid_values, expected_pid)
    )


def test_export_fails_for_non_monotonic_time(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "scored.csv"
    out_path = tmp_path / "ref.mat"
    meta_path = tmp_path / "meta.json"
    write_csv(
        csv_path,
        [
            ",".join(HEADER),
            make_row(
                "2024-01-01T00:00:01Z",
                "u0",
                "s0",
                "catA",
                "0.0",
                "0.1",
                "0.2",
                "0.3",
                "1.0",
                "1",
                "0",
            ),
            make_row(
                "2024-01-01T00:00:00Z",
                "u0",
                "s0",
                "catA",
                "0.0",
                "0.1",
                "0.2",
                "0.3",
                "0.5",
                "0",
                "1",
            ),
        ],
    )

    expected_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    exit_code = cli.main(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
            "--seed",
            "5",
        ]
    )

    assert exit_code == 1
    assert not out_path.exists()
    assert not meta_path.exists()

    stdout = capsys.readouterr().out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    assert logs[0]["event"] == "export.start"
    assert logs[0]["seed"] == 5
    assert logs[0]["input_sha256"] == expected_sha
    assert logs[-1]["event"] == "export.error"
    assert logs[-1]["seed"] == 5
    assert logs[-1]["input_sha256"] == expected_sha


def test_export_missing_seed_fails(tmp_path: Path, capsys) -> None:
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
                "0.2",
                "0.3",
                "0.9",
                "0",
                "0",
            ),
        ],
    )

    exit_code = cli.main(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
        ]
    )

    assert exit_code == 1
    assert not out_path.exists()
    assert not meta_path.exists()

    stdout = capsys.readouterr().out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    assert logs[0]["event"] == "export.error"
    assert logs[0]["error"] == "missing_seed"


def test_export_missing_meta_fails(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "scored.csv"
    out_path = tmp_path / "ref.mat"
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
                "0.2",
                "0.3",
                "0.9",
                "0",
                "0",
            ),
        ],
    )

    exit_code = cli.main(
        ["export", "--in", str(csv_path), "--out", str(out_path), "--seed", "9"]
    )

    assert exit_code == 1
    assert not out_path.exists()

    stdout = capsys.readouterr().out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    assert logs[0]["event"] == "export.error"
    assert logs[0]["error"] == "missing_meta"


def test_export_fails_when_header_mismatch(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "scored.csv"
    out_path = tmp_path / "ref.mat"
    meta_path = tmp_path / "meta.json"
    invalid_header = HEADER[:-1]
    write_csv(
        csv_path,
        [
            ",".join(invalid_header),
            "2024-01-01T00:00:00Z,u0,s0,catA,0.0,0.1,0.2,0.3,0.9,0",
        ],
    )

    expected_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    exit_code = cli.main(
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
        ]
    )

    assert exit_code == 1
    stdout = capsys.readouterr().out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    assert logs[0]["event"] == "export.start"
    assert logs[0]["input_sha256"] == expected_sha
    assert logs[-1]["event"] == "export.error"
    assert logs[-1]["error"] == "validation_error"
    assert logs[-1]["input_sha256"] == expected_sha
    assert "header" in logs[-1]["message"].lower()


def test_export_fails_on_non_numeric_signal(tmp_path: Path, capsys) -> None:
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
                "0.2",
                "not-a-number",
                "0.5",
                "0",
                "0",
            ),
        ],
    )

    expected_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    exit_code = cli.main(
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
        ]
    )

    assert exit_code == 1
    stdout = capsys.readouterr().out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    assert logs[0]["event"] == "export.start"
    assert logs[0]["input_sha256"] == expected_sha
    assert logs[-1]["event"] == "export.error"
    assert logs[-1]["error"] == "validation_error"
    assert logs[-1]["input_sha256"] == expected_sha
    assert "s_time" in logs[-1]["message"]


def test_export_missing_input_file_logs_none_hash(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "missing.csv"
    out_path = tmp_path / "ref.mat"
    meta_path = tmp_path / "meta.json"

    exit_code = cli.main(
        [
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(out_path),
            "--meta",
            str(meta_path),
            "--seed",
            "99",
        ]
    )

    assert exit_code == 1
    assert not out_path.exists()
    assert not meta_path.exists()

    stdout = capsys.readouterr().out.strip().splitlines()
    logs = [json.loads(line) for line in stdout if line]
    assert logs[0]["event"] == "export.start"
    assert logs[0]["seed"] == 99
    assert logs[0]["input_sha256"] is None
    assert logs[-1]["event"] == "export.error"
    assert logs[-1]["error"] == "file_not_found"
    assert logs[-1]["input_sha256"] is None
