from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from log_generator.generate import (
    OperationSpec,
    ScenarioSpec,
    Transition,
    TokenReplayAnomalySpec,
    UnauthAnomalySpec,
    UserSpec,
    generate_anom_records,
)


def _constant_cdf(seconds: float) -> list[tuple[float, float]]:
    return [(0.0, seconds), (1.0, seconds)]


def _spec_with_unauth(probability: float) -> ScenarioSpec:
    operations = {
        "login": OperationSpec(
            method="POST",
            path="/login",
            referer="-",
            user_agent="pytest",
            ip="127.0.0.1",
            op_category="AUTH",
            transitions=[Transition(op="update", prob=1.0)],
            cdf_points=_constant_cdf(5.0),
        ),
        "update": OperationSpec(
            method="POST",
            path="/api/update",
            referer="/login",
            user_agent="pytest",
            ip="127.0.0.1",
            op_category="UPDATE",
            transitions=[Transition(op="update", prob=1.0)],
            cdf_points=_constant_cdf(5.0),
        ),
    }

    return ScenarioSpec(
        algo_version="test-unauth",
        t0=datetime(1970, 1, 1, tzinfo=timezone.utc),
        users=[
            UserSpec(
                uid="user-1",
                session_id="sess-1",
                initial_op="login",
                steps=2,
            )
        ],
        operations=operations,
        time_anomalies=[],
        order_anomalies=[],
        unauth_anomalies=[UnauthAnomalySpec(probability=probability, op=None)],
        token_replay_anomalies=[],
    )


def _spec_with_token(probability: float) -> ScenarioSpec:
    operations = {
        "view": OperationSpec(
            method="GET",
            path="/view",
            referer="-",
            user_agent="pytest",
            ip="127.0.0.1",
            op_category="READ",
            transitions=[Transition(op="view", prob=1.0)],
            cdf_points=_constant_cdf(5.0),
        ),
    }

    return ScenarioSpec(
        algo_version="test-token",
        t0=datetime(1970, 1, 1, tzinfo=timezone.utc),
        users=[
            UserSpec(
                uid="user-1",
                session_id="sess-1",
                initial_op="view",
                steps=1,
            ),
            UserSpec(
                uid="user-2",
                session_id="sess-2",
                initial_op="view",
                steps=1,
            ),
        ],
        operations=operations,
        time_anomalies=[],
        order_anomalies=[],
        unauth_anomalies=[],
        token_replay_anomalies=[
            TokenReplayAnomalySpec(probability=probability, op=None)
        ],
    )


def test_unauth_anomaly_breaks_auth_predicate_and_logs_audit() -> None:
    spec = _spec_with_unauth(probability=1.0)

    records, audit_entries = generate_anom_records(spec, seed=11)

    assert len(records) == 2
    login_record, update_record = records
    assert login_record["session_id"] == "sess-1"
    assert update_record["session_id"] != "sess-1"
    assert "unauth" in update_record["session_id"].lower()

    unauth_entries = [entry for entry in audit_entries if entry["type"] == "unauth"]
    assert len(unauth_entries) == 1
    entry = unauth_entries[0]
    assert entry["record_index"] == 1
    assert entry["session_original"] == "sess-1"
    assert entry["session_mutated"] == update_record["session_id"]
    assert "AuthOK" in entry["reason"]


def test_token_replay_anomaly_reuses_foreign_session_and_is_deterministic() -> None:
    spec = _spec_with_token(probability=1.0)

    records_first, audit_first = generate_anom_records(spec, seed=21)
    records_second, audit_second = generate_anom_records(spec, seed=21)

    assert records_first == records_second
    assert audit_first == audit_second

    session_map = {row["uid"]: row["session_id"] for row in records_first}
    assert session_map == {"user-1": "sess-2", "user-2": "sess-1"}

    token_entries = [entry for entry in audit_first if entry["type"] == "token_replay"]
    assert len(token_entries) == 2
    indexed = {entry["record_index"]: entry for entry in token_entries}
    assert indexed[0]["session_original"] == "sess-1"
    assert indexed[0]["session_replayed"] == "sess-2"
    assert indexed[0]["token_owner_uid"] == "user-2"
    assert "uid×session_id" in indexed[0]["reason"]
    assert indexed[1]["session_original"] == "sess-2"
    assert indexed[1]["session_replayed"] == "sess-1"
    assert indexed[1]["token_owner_uid"] == "user-1"
