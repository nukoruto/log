"""Tests for time anomaly injection semantics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from log_generator.generate import (
    OperationSpec,
    ScenarioSpec,
    TimeAnomalySpec,
    Transition,
    UserSpec,
    generate_anom_records,
)


def _constant_cdf(seconds: float) -> list[tuple[float, float]]:
    return [(0.0, seconds), (1.0, seconds)]


def _base_spec(time_anomaly: TimeAnomalySpec) -> ScenarioSpec:
    operations = {
        "start": OperationSpec(
            method="GET",
            path="/start",
            referer="-",
            user_agent="pytest",
            ip="127.0.0.1",
            op_category="start",
            transitions=[Transition(op="stay", prob=1.0)],
            cdf_points=_constant_cdf(10.0),
        ),
        "stay": OperationSpec(
            method="GET",
            path="/stay",
            referer="/start",
            user_agent="pytest",
            ip="127.0.0.1",
            op_category="stay",
            transitions=[Transition(op="stay", prob=1.0)],
            cdf_points=_constant_cdf(10.0),
        ),
    }

    return ScenarioSpec(
        algo_version="test",
        t0=datetime(1970, 1, 1, tzinfo=timezone.utc),
        users=[
            UserSpec(
                uid="user",
                session_id="sess",
                initial_op="start",
                steps=3,
            )
        ],
        operations=operations,
        time_anomalies=[time_anomaly],
        order_anomalies=[],
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_local_time_anomaly_affects_only_target_op() -> None:
    spec = _base_spec(
        TimeAnomalySpec(
            mode="local",
            probability=1.0,
            op="start",
            scale=0.5,
            delta=None,
        )
    )

    records, audit_entries = generate_anom_records(spec, seed=42)

    base = spec.t0
    times = [_parse_timestamp(row["timestamp_utc"]) for row in records]
    assert times == [
        base,
        base + timedelta(seconds=5),
        base + timedelta(seconds=15),
    ]

    assert len(audit_entries) == 1
    entry = audit_entries[0]
    assert entry["type"] == "time"
    assert entry["mode"] == "local"
    assert entry["record_index"] == 0
    assert entry["scale"] == pytest.approx(0.5)
    assert entry["params"]["scale"] == pytest.approx(0.5)
    assert "delta" not in entry["params"]
    assert "reason" in entry and "local" in entry["reason"]


def test_propagate_time_anomaly_reintegrates_future_events() -> None:
    spec = _base_spec(
        TimeAnomalySpec(
            mode="propagate",
            probability=1.0,
            op="start",
            scale=0.5,
            delta=None,
        )
    )

    records, audit_entries = generate_anom_records(spec, seed=42)

    base = spec.t0
    times = [_parse_timestamp(row["timestamp_utc"]) for row in records]
    assert times == [
        base,
        base + timedelta(seconds=5),
        base + timedelta(seconds=10),
    ]

    assert len(audit_entries) == 1
    entry = audit_entries[0]
    assert entry["type"] == "time"
    assert entry["mode"] == "propagate"
    assert entry["record_index"] == 0
    assert entry["scale"] == pytest.approx(0.5)
    assert entry["params"]["scale"] == pytest.approx(0.5)
    assert entry["seconds_before"] == pytest.approx(10.0)
    assert entry["seconds_after"] == pytest.approx(5.0)
    assert "reason" in entry and "propagate" in entry["reason"]
    assert "C.2" in entry["reason"]


def test_time_anomaly_generation_is_seed_deterministic() -> None:
    spec = _base_spec(
        TimeAnomalySpec(
            mode="local",
            probability=0.5,
            op="start",
            scale=0.5,
            delta=3.0,
        )
    )

    records_first, audit_first = generate_anom_records(spec, seed=123)
    records_second, audit_second = generate_anom_records(spec, seed=123)

    assert records_first == records_second
    assert audit_first == audit_second
    if audit_first:
        params = audit_first[0]["params"]
        assert params["scale"] == pytest.approx(0.5)
        assert params["delta"] == pytest.approx(3.0)
