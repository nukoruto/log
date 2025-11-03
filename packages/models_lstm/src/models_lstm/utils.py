"""Utility helpers for deterministic seeding and device resolution."""

from __future__ import annotations

import os
import random
import re
from typing import Final

import numpy  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]

_KNOWN_GPU_MODES: Final[frozenset[str]] = frozenset({"rtx6000", "rtx4060", "cpu"})


def set_deterministic_mode(seed: int) -> None:
    """Seed RNGs and enable deterministic backend features."""

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no branch - depends on runtime
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():  # pragma: no branch
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:  # pragma: no cover - deterministic backend unavailable
        pass


def _normalize_gpu_label(label: str) -> str:
    """Return a normalized key for GPU name comparisons."""

    return re.sub(r"[^a-z0-9]", "", label.lower())


def resolve_device() -> torch.device:
    """Resolve the preferred ``torch.device`` from the GPU_MODE environment."""

    mode = os.environ.get("GPU_MODE", "cpu").strip().lower()
    if mode not in _KNOWN_GPU_MODES:
        return torch.device("cpu")
    if mode == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    target_index: int | None = None
    device_count = torch.cuda.device_count()
    normalized_mode = _normalize_gpu_label(mode)
    for index in range(device_count):
        name = torch.cuda.get_device_name(index)
        normalized_name = _normalize_gpu_label(name)
        if normalized_mode and normalized_mode in normalized_name:
            target_index = index
            break

    if target_index is None:
        target_index = 0 if device_count > 0 else None

    if target_index is None:
        return torch.device("cpu")

    return torch.device(f"cuda:{target_index}")
