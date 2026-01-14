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

    mode = os.environ.get("GPU_MODE", "").strip().lower()
    
    # If explicit CPU requested
    if mode == "cpu":
        return torch.device("cpu")

    # If specific GPU requested (rtx6000, rtx4060)
    if mode in _KNOWN_GPU_MODES and mode != "cpu":
        # Keep existing logic for specific targeting if needed, 
        # or just fall through to general CUDA check if we treat them as "use best available"
        # For now, let's keep the target index logic for specific modes
        pass
    else:
        # Auto mode (default): use CUDA if available
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    target_index: int | None = None
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        normalized_mode = _normalize_gpu_label(mode)
        for index in range(device_count):
            name = torch.cuda.get_device_name(index)
            normalized_name = _normalize_gpu_label(name)
            if normalized_mode and normalized_mode in normalized_name:
                target_index = index
                break
        
        if target_index is None and device_count > 0:
             # Fallback to first device if specific name not found but mode was valid (shouldn't happen with strict check but good for safety)
             target_index = 0

    if target_index is None:
        return torch.device("cpu")

    return torch.device(f"cuda:{target_index}")
