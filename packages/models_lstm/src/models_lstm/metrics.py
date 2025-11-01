"""Utility helpers for computing and persisting evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

__all__ = [
    "compute_linear_quantile",
    "build_category_thresholds",
    "save_metrics",
]

_REQUIRED_FIELDS: Sequence[str] = (
    "f1",
    "pr_auc",
    "roc_auc",
    "detection_delay",
    "thresholds",
)


def compute_linear_quantile(values: Sequence[float], quantile: float) -> float:
    """Compute a quantile using linear interpolation.

    Args:
        values: Observations from which to compute the quantile.
        quantile: Quantile value in the inclusive range [0, 1].

    Returns:
        The linearly interpolated quantile value.

    Raises:
        ValueError: If ``values`` is empty or ``quantile`` is outside [0, 1].
    """

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")

    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = quantile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return float(lower_value + (upper_value - lower_value) * fraction)


def build_category_thresholds(
    residuals_by_category: Mapping[str, Sequence[float]],
    categories: Sequence[str],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Create per-category quantile thresholds using linear interpolation.

    Args:
        residuals_by_category: Mapping from category to residual samples.
        categories: Ordered categories to emit in the result.
        alpha: Lower-tail quantile level; upper level uses ``1 - alpha``.

    Returns:
        Dictionary describing the quantile thresholds per category.

    Raises:
        ValueError: If ``alpha`` lies outside the inclusive range [0, 0.5].
    """

    if not 0.0 <= alpha <= 0.5:
        raise ValueError("alpha must be between 0 and 0.5")

    per_category: Dict[str, Dict[str, Optional[float]]] = {}
    for category in categories:
        residuals = [float(value) for value in residuals_by_category.get(category, ())]
        if residuals:
            tau_lo = compute_linear_quantile(residuals, alpha)
            tau_hi = compute_linear_quantile(residuals, 1.0 - alpha)
        else:
            tau_lo = None
            tau_hi = None
        per_category[category] = {"tau_lo": tau_lo, "tau_hi": tau_hi}

    return {
        "alpha": float(alpha),
        "strategy": "quantile",
        "method": "linear",
        "per_category": per_category,
    }


def save_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist evaluation metrics to JSON, enforcing required keys.

    Args:
        path: Destination for the metrics JSON file.
        payload: Mapping that must include the core metric fields.

    Raises:
        ValueError: If required fields are absent or ``thresholds`` is invalid.
    """

    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            "metrics payload is missing required fields: " + ", ".join(sorted(missing))
        )

    thresholds = payload["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must be a mapping")

    # Convert payload to a JSON-serialisable dictionary.
    serialisable: Dict[str, Any] = {
        key: _to_serialisable(value) for key, value in dict(payload).items()
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialisable, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _to_serialisable(value: Any) -> Any:
    """Convert a value into a JSON-serialisable structure.

    Args:
        value: Arbitrary object to convert.

    Returns:
        JSON-compatible representation of ``value``.
    """

    if isinstance(value, Mapping):
        return {key: _to_serialisable(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_serialisable(item) for item in value]
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return value
