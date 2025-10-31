"""Unit tests for metrics utilities in models_lstm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models_lstm.metrics import (  # noqa: E402  # type: ignore[attr-defined]
    build_category_thresholds,
    compute_linear_quantile,
    save_metrics,
)


def test_compute_linear_quantile_interpolates_between_points() -> None:
    values = [0.0, 1.0, 3.0, 9.0]
    assert compute_linear_quantile(values, 0.25) == pytest.approx(0.75)
    assert compute_linear_quantile(values, 0.5) == pytest.approx(2.0)
    assert compute_linear_quantile(values, 1.0) == pytest.approx(9.0)


def test_build_category_thresholds_uses_linear_quantiles() -> None:
    residuals = {
        "cat_a": [0.1, 0.2, 0.3, 0.4],
        "cat_b": [1.0, 1.5],
    }
    categories = ["cat_a", "cat_b", "cat_c"]

    thresholds = build_category_thresholds(residuals, categories, alpha=0.25)

    assert thresholds["method"] == "linear"
    assert thresholds["strategy"] == "quantile"

    per_category = thresholds["per_category"]
    assert per_category["cat_a"]["tau_lo"] == pytest.approx(0.175)
    assert per_category["cat_a"]["tau_hi"] == pytest.approx(0.325)
    assert per_category["cat_b"]["tau_lo"] == pytest.approx(1.125)
    assert per_category["cat_b"]["tau_hi"] == pytest.approx(1.375)
    assert per_category["cat_c"]["tau_lo"] is None
    assert per_category["cat_c"]["tau_hi"] is None


def test_save_metrics_persists_required_fields(tmp_path: Path) -> None:
    payload = {
        "f1": 0.81,
        "pr_auc": 0.92,
        "roc_auc": 0.88,
        "detection_delay": 4.5,
        "thresholds": {
            "method": "linear",
            "strategy": "quantile",
            "per_category": {},
        },
        "history": [
            {"epoch": 1.0, "train_loss": 0.5, "val_loss": 0.6},
        ],
    }

    output_path = tmp_path / "metrics.json"
    save_metrics(output_path, payload)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    for key in ("f1", "pr_auc", "roc_auc", "detection_delay", "thresholds"):
        assert key in saved
    assert saved["thresholds"]["method"] == "linear"


@pytest.mark.parametrize(
    "missing_key",
    ["f1", "pr_auc", "roc_auc", "detection_delay", "thresholds"],
)
def test_save_metrics_requires_core_fields(tmp_path: Path, missing_key: str) -> None:
    payload = {
        "f1": 0.1,
        "pr_auc": 0.2,
        "roc_auc": 0.3,
        "detection_delay": 0.4,
        "thresholds": {"method": "linear", "strategy": "quantile", "per_category": {}},
    }
    payload.pop(missing_key)

    with pytest.raises(ValueError):
        save_metrics(tmp_path / "metrics.json", payload)
