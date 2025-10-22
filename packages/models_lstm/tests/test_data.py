import math
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models_lstm.data import (  # noqa: E402
    FeaturePipelineResult,
    prepare_lstm_features,
    split_group_kfold,
)


@pytest.fixture()
def sample_contract_rows() -> List[Dict[str, Any]]:
    return [
        {
            "timestamp_utc": "1970-01-01T00:00:00Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "GET",
            "path": "/login",
            "referer": "-",
            "user_agent": "ua",
            "ip": "0.0.0.0",
            "op_category": "login",
        },
        {
            "timestamp_utc": "1970-01-01T00:00:10Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "POST",
            "path": "/login",
            "referer": "/login",
            "user_agent": "ua",
            "ip": "0.0.0.0",
            "op_category": "login",
        },
        {
            "timestamp_utc": "1970-01-01T00:10:10Z",
            "uid": "u1",
            "session_id": "s2",
            "method": "GET",
            "path": "/dashboard",
            "referer": "/login",
            "user_agent": "ua",
            "ip": "0.0.0.0",
            "op_category": "dashboard",
        },
        {
            "timestamp_utc": "1970-01-01T01:10:10Z",
            "uid": "u2",
            "session_id": "s3",
            "method": "GET",
            "path": "/",
            "referer": "-",
            "user_agent": "ua",
            "ip": "0.0.0.0",
            "op_category": "home",
        },
        {
            "timestamp_utc": "1970-01-01T01:10:20Z",
            "uid": "u2",
            "session_id": "s3",
            "method": "GET",
            "path": "/reports",
            "referer": "/",
            "user_agent": "ua",
            "ip": "0.0.0.0",
            "op_category": "report",
        },
        {
            "timestamp_utc": "1970-01-01T02:10:20Z",
            "uid": "u2",
            "session_id": "s4",
            "method": "POST",
            "path": "/upload",
            "referer": "/reports",
            "user_agent": "ua",
            "ip": "0.0.0.0",
            "op_category": "upload",
        },
    ]


def compute_expected_z_scores(rows: List[Dict[str, Any]]) -> List[float]:
    parsed: List[Dict[str, Any]] = [dict(item) for item in rows]
    for row in parsed:
        ts = str(row["timestamp_utc"]).replace("Z", "+00:00")
        row["timestamp_utc"] = datetime.fromisoformat(ts)
    parsed.sort(key=lambda r: (r["uid"], r["session_id"], r["timestamp_utc"]))

    log_deltas: List[float] = []
    prev_times: Dict[str, datetime] = {}
    for row in parsed:
        key = f"{row['uid']}::{row['session_id']}"
        prev_time = prev_times.get(key)
        if prev_time is None:
            delta_seconds = 0.0
        else:
            delta = row["timestamp_utc"] - prev_time
            delta_seconds = delta.total_seconds()
        log_deltas.append(math.log(delta_seconds + 1e-3))
        prev_times[key] = row["timestamp_utc"]

    median = statistics.median(log_deltas)
    mad = statistics.median([abs(x - median) for x in log_deltas])
    scale = 1.4826 * (mad if mad > 0 else 1.0)
    z_scores = [(value - median) / scale for value in log_deltas]
    return [max(min(z, 5.0), -5.0) for z in z_scores]


def test_prepare_lstm_features_creates_expected_columns(
    sample_contract_rows: List[Dict[str, Any]],
) -> None:
    result = prepare_lstm_features(sample_contract_rows)
    assert isinstance(result, FeaturePipelineResult)
    feature_rows = result.features

    op_categories = [row["op_category"] for row in feature_rows]
    op_indices = [row["op_index"] for row in feature_rows]
    mapped_indices = [result.op_index_mapping[cat] for cat in op_categories]
    assert op_indices == mapped_indices
    assert sorted(result.op_index_mapping.keys()) == [
        "dashboard",
        "home",
        "login",
        "report",
        "upload",
    ]

    expected_z = compute_expected_z_scores(sample_contract_rows)
    actual_z = [row["z_clip"] for row in feature_rows]
    for computed, expected in zip(actual_z, expected_z):
        assert computed == pytest.approx(expected, abs=1e-6)

    stats = asdict(result.stats)
    assert "log_median" in stats and "log_mad" in stats
    assert stats["clip_min"] == pytest.approx(-5.0)
    assert stats["clip_max"] == pytest.approx(5.0)


def test_prepare_lstm_features_missing_column_raises(
    sample_contract_rows: List[Dict[str, Any]],
) -> None:
    broken = [dict(row) for row in sample_contract_rows]
    broken[0].pop("referer")
    with pytest.raises(ValueError, match="referer"):
        prepare_lstm_features(broken)


def test_split_group_kfold_respects_session_boundaries(
    sample_contract_rows: List[Dict[str, Any]],
) -> None:
    result = prepare_lstm_features(sample_contract_rows)
    splits = list(split_group_kfold(result.features, n_splits=2))

    assert len(splits) == 2
    group_labels = [f"{row['uid']}::{row['session_id']}" for row in result.features]
    for train_idx, val_idx in splits:
        train_groups = {group_labels[i] for i in train_idx}
        val_groups = {group_labels[i] for i in val_idx}
        assert train_groups.isdisjoint(val_groups)
        assert len(val_idx) > 0
        assert len(train_idx) > 0
