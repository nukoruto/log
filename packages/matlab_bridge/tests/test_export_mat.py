"""Tests for MATLAB bridge export."""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from typing import Dict, List

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _install_minimal_scipy_loader() -> None:
    """Provide a lightweight scipy.io.loadmat implementation for tests."""

    if "scipy.io" in sys.modules:
        return

    MI_MATRIX = 14

    def _padded_length(length: int) -> int:
        remainder = length % 8
        return length if remainder == 0 else length + (8 - remainder)

    def loadmat(path: str | Path) -> Dict[str, List[List[float]]]:
        with Path(path).open("rb") as handle:
            blob = handle.read()

        offset = 128  # skip header
        result: Dict[str, List[List[float]]] = {}

        while offset + 8 <= len(blob):
            mi_type, num_bytes = struct.unpack_from("<II", blob, offset)
            offset += 8
            padded = _padded_length(num_bytes)
            payload = blob[offset : offset + num_bytes]
            offset += padded

            if mi_type != MI_MATRIX:
                continue

            cursor = 0

            def read_element() -> tuple[int, bytes]:
                nonlocal cursor
                tag_type, tag_bytes = struct.unpack_from("<II", payload, cursor)
                cursor += 8
                data = payload[cursor : cursor + tag_bytes]
                cursor += _padded_length(tag_bytes)
                return tag_type, data

            _, _flags = read_element()
            _, dims_data = read_element()
            rows, cols = struct.unpack("<ii", dims_data)
            _, name_data = read_element()
            name = name_data.decode("utf-8").rstrip("\x00")
            _, real_data = read_element()

            values: List[float] = [
                struct.unpack_from("<d", real_data, idx)[0]
                for idx in range(0, len(real_data), 8)
            ]

            column_major: List[List[float]] = [values[row::rows] for row in range(rows)]
            matrix: List[List[float]] = [
                [column_major[row][col] for col in range(cols)] for row in range(rows)
            ]
            result[name] = matrix

        return result

    scipy_module = types.ModuleType("scipy")
    io_module = types.ModuleType("scipy.io")
    setattr(io_module, "loadmat", loadmat)
    setattr(scipy_module, "io", io_module)
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.io"] = io_module


_install_minimal_scipy_loader()

from matlab_bridge import cli  # noqa: E402


def write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("t,ref,y_lstm,y_pid\n")
        for row in rows:
            fh.write(",".join(row) + "\n")


def test_export_creates_mat_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "scored.csv"
    write_csv(csv_path, [("0", "0", "0", "0"), ("1", "1", "0.8", "0.9")])
    out_path = tmp_path / "ref.mat"

    exit_code = cli.main(
        ["export", "--in", str(csv_path), "--out", str(out_path), "--seed", "7"]
    )
    assert exit_code == 0

    output_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.strip().splitlines()
        if line.strip()
    ]
    assert output_lines[0]["event"] == "start"
    assert output_lines[-1]["event"] == "complete"

    from scipy.io import loadmat  # type: ignore[attr-defined, import-not-found, import-untyped]

    data = loadmat(out_path)
    for key in ("ref", "y_lstm", "y_pid", "t"):
        assert key in data
        assert len(data[key]) == 2
        assert len(data[key][0]) == 1


def test_missing_seed_raises_system_exit(tmp_path: Path) -> None:
    csv_path = tmp_path / "scored.csv"
    write_csv(csv_path, [("0", "0", "0", "0"), ("1", "1", "1", "1")])
    out_path = tmp_path / "ref.mat"

    with pytest.raises(SystemExit):
        cli.main(["export", "--in", str(csv_path), "--out", str(out_path)])


def test_non_monotonic_time_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "scored.csv"
    write_csv(csv_path, [("0", "0", "0", "0"), ("0", "1", "1", "1")])
    out_path = tmp_path / "ref.mat"

    exit_code = cli.main(
        ["export", "--in", str(csv_path), "--out", str(out_path), "--seed", "11"]
    )
    assert exit_code == 1

    logs = [
        json.loads(line)
        for line in capsys.readouterr().out.strip().splitlines()
        if line.strip()
    ]
    assert logs[-1]["event"] == "error"
