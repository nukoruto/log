"""Tests covering device resolution and RNG seeding helpers."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("numpy")
pytest.importorskip("torch")

from models_lstm import utils as utils_module  # noqa: E402


def test_resolve_device_returns_cpu_for_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU_MODE が未定義値の場合は CPU を選択する。"""

    monkeypatch.setenv("GPU_MODE", "mystery_gpu")
    monkeypatch.setattr(utils_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils_module.torch.cuda, "device_count", lambda: 2)

    def _get_device_name(index: int) -> str:
        names = ["nvidia a100", "nvidia tesla"]
        return names[index]

    monkeypatch.setattr(utils_module.torch.cuda, "get_device_name", _get_device_name)

    device = utils_module.resolve_device()

    assert device.type == "cpu"


def test_resolve_device_prefers_named_gpu_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既知の GPU 名が存在しない場合は cuda:0 を選択する。"""

    monkeypatch.setenv("GPU_MODE", "rtx4060")
    monkeypatch.setattr(utils_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils_module.torch.cuda, "device_count", lambda: 2)

    names: List[str] = ["nvidia rtx3090", "nvidia tesla"]
    monkeypatch.setattr(
        utils_module.torch.cuda,
        "get_device_name",
        lambda index: names[index],
    )

    device = utils_module.resolve_device()

    assert device.type == "cuda"
    assert device.index == 0


def test_resolve_device_matches_requested_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """指定された GPU 名に合致するデバイスを優先する。"""

    monkeypatch.setenv("GPU_MODE", "rtx6000")
    monkeypatch.setattr(utils_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils_module.torch.cuda, "device_count", lambda: 2)

    names: List[str] = ["nvidia tesla", "nvidia rtx6000"]
    monkeypatch.setattr(
        utils_module.torch.cuda,
        "get_device_name",
        lambda index: names[index],
    )

    device = utils_module.resolve_device()

    assert device.type == "cuda"
    assert device.index == 1


def test_resolve_device_matches_gpu_name_with_spacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU 名のスペースや記号差異を吸収してマッチさせる。"""

    monkeypatch.setenv("GPU_MODE", "rtx6000")
    monkeypatch.setattr(utils_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils_module.torch.cuda, "device_count", lambda: 2)

    names: List[str] = ["nvidia tesla", "NVIDIA RTX 6000 Ada Generation"]
    monkeypatch.setattr(
        utils_module.torch.cuda,
        "get_device_name",
        lambda index: names[index],
    )

    device = utils_module.resolve_device()

    assert device.type == "cuda"
    assert device.index == 1


def test_resolve_device_uses_cpu_when_cuda_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA 非搭載環境では常に CPU を返す。"""

    monkeypatch.setenv("GPU_MODE", "rtx6000")
    monkeypatch.setattr(utils_module.torch.cuda, "is_available", lambda: False)

    device = utils_module.resolve_device()

    assert device.type == "cpu"


def test_set_deterministic_mode_resets_rngs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed を再設定すると主要 RNG が再現可能になる。"""

    import numpy  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    monkeypatch.setattr(utils_module.torch.cuda, "is_available", lambda: False)

    utils_module.set_deterministic_mode(123)
    baseline_random = random.random()
    baseline_numpy = float(numpy.random.rand())
    baseline_torch = float(torch.rand(1).item())

    utils_module.set_deterministic_mode(123)
    repeated_random = random.random()
    repeated_numpy = float(numpy.random.rand())
    repeated_torch = float(torch.rand(1).item())

    assert baseline_random == pytest.approx(repeated_random)
    assert baseline_numpy == pytest.approx(repeated_numpy)
    assert baseline_torch == pytest.approx(repeated_torch)
