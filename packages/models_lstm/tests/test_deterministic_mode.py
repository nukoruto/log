"""Tests for deterministic seeding utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("numpy")
pytest.importorskip("torch")

from models_lstm import train  # noqa: E402


def test_set_deterministic_mode_skips_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CUDA-specific seeding is skipped when CUDA is unavailable."""

    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    def _manual_seed_all(_: int) -> None:
        raise AssertionError("CUDA seeding should be skipped when CUDA is unavailable")

    monkeypatch.setattr(train.torch.cuda, "manual_seed_all", _manual_seed_all)

    # Simulate cudnn being unavailable to guard attribute access.
    cudnn_module = train.torch.backends.cudnn
    monkeypatch.setattr(cudnn_module, "is_available", lambda: False)

    train.set_deterministic_mode(123)
