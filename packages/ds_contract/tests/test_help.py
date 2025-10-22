"""Tests for the ds_contract CLI skeleton."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "packages" / "ds_contract" / "src"


@pytest.mark.usefixtures("tmp_path")
def test_cli_help() -> None:
    """`ds-contract --help` should exit successfully and display help text."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-m", "ds_contract.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: ds-contract" in result.stdout
    assert "Tools for validating" in result.stdout
