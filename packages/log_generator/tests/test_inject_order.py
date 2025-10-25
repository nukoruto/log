from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from log_generator.generate import (
    OperationSpec,
    OrderAnomalySpec,
    ScenarioSpec,
    Transition,
    UserSpec,
    generate_anom_records,
    generate_normal_records,
)


def _constant_cdf(seconds: float) -> list[tuple[float, float]]:
    return [(0.0, seconds), (1.0, seconds)]


def _order_spec(probability: float) -> ScenarioSpec:
    operations = {
        "login": OperationSpec(
            method="POST",
            path="/login",
            referer="-",
            user_agent="pytest",
            ip="127.0.0.1",
            op_category="auth",
            transitions=[Transition(op="confirm", prob=1.0)],
            cdf_points=_constant_cdf(5.0),
        ),
        "confirm": OperationSpec(
            method="GET",
            path="/confirm",
            referer="/login",
            user_agent="pytest",
            ip="127.0.0.1",
            op_category="auth-confirm",
            transitions=[Transition(op="confirm", prob=1.0)],
            cdf_points=_constant_cdf(5.0),
        ),
    }

    return ScenarioSpec(
        algo_version="test-order",
        t0=datetime(1970, 1, 1, tzinfo=timezone.utc),
        users=[
            UserSpec(
                uid="user",
                session_id="sess",
                initial_op="login",
                steps=2,
            )
        ],
        operations=operations,
        time_anomalies=[],
        order_anomalies=[OrderAnomalySpec(probability=probability, op="login")],
    )


def test_order_anomaly_swaps_operations_and_logs_audit() -> None:
    spec = _order_spec(probability=1.0)

    records, audit_entries = generate_anom_records(spec, seed=7)

    assert [row["path"] for row in records] == ["/confirm", "/login"]

    assert len(audit_entries) == 1
    entry = audit_entries[0]
    assert entry["type"] == "order"
    assert entry["record_index"] == 0
    assert entry["swap_with"] == 1
    assert entry["op_before"] == "login"
    assert entry["op_after"] == "confirm"
    assert entry["swap_op_before"] == "confirm"
    assert entry["swap_op_after"] == "login"
    assert entry["constraint"] == "login->confirm"
    reason = entry["reason"]
    assert "ℛ" in reason
    assert "order constraint" in reason


def test_order_anomaly_not_injected_into_normal_records() -> None:
    spec = _order_spec(probability=1.0)

    normal_records = generate_normal_records(spec, seed=7)

    assert [row["path"] for row in normal_records] == ["/login", "/confirm"]
