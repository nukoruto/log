"""End-to-end tests for matlab-bridge export CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .mat_stub import loadmat


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


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_row(
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


ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "packages" / "matlab_bridge" / "src"


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SRC_DIR)]
    if existing := env.get("PYTHONPATH"):
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_cli_export_twice_with_consistent_shapes(tmp_path: Path) -> None:
    csv_path = tmp_path / "scored.csv"
    _write_csv(
        csv_path,
        [
            ",".join(HEADER),
            _make_row(
                "2024-01-01T00:00:00Z",
                "u0",
                "s0",
                "catA",
                "0.0",
                "0.1",
                "0.01",
                "0.00",
                "0.5",
                "0",
                "0",
            ),
            _make_row(
                "2024-01-01T00:00:01Z",
                "u0",
                "s0",
                "catA",
                "0.1",
                "0.2",
                "0.02",
                "0.50",
                "0.7",
                "0",
                "1",
            ),
            _make_row(
                "2024-01-01T00:00:02Z",
                "u0",
                "s0",
                "catA",
                "0.2",
                "0.3",
                "0.03",
                "1.10",
                "0.9",
                "1",
                "0",
            ),
            _make_row(
                "2024-01-01T00:00:03Z",
                "u0",
                "s0",
                "catA",
                "0.3",
                "0.4",
                "0.04",
                "1.80",
                "1.1",
                "1",
                "1",
            ),
        ],
    )

    def run_once(output_dir: Path) -> tuple[dict[str, list[float]], tuple[int, int]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_path = output_dir / "ref.mat"
        meta_path = output_dir / "meta.json"
        command = [
            sys.executable,
            "-m",
            "matlab_bridge.cli",
            "export",
            "--in",
            str(csv_path),
            "--out",
            str(ref_path),
            "--meta",
            str(meta_path),
            "--seed",
            "101",
        ]
        result = _run_cli(command)
        assert result.returncode == 0, result.stdout + result.stderr

        logs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert logs[0]["event"] == "export.start"
        assert logs[-1]["event"] == "export.complete"

        mat = loadmat(ref_path)
        for key in ("ref", "y_lstm", "y_pid", "t"):
            assert key in mat

        shapes = {mat[key].shape for key in ("ref", "y_lstm", "y_pid", "t")}
        assert len(shapes) == 1
        (shape,) = tuple(shapes)
        assert shape[0] * shape[1] == 4

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["row_count"] == 4

        flattened = {key: mat[key].flatten() for key in ("ref", "y_lstm", "y_pid", "t")}
        return flattened, shape

    first_values, first_shape = run_once(tmp_path / "run1")
    second_values, second_shape = run_once(tmp_path / "run2")

    assert first_shape == second_shape
    assert first_values == second_values
