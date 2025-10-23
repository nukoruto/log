"""Core generation utilities implementing the scenario specification contract."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

CONTRACT_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "uid",
    "session_id",
    "method",
    "path",
    "referer",
    "user_agent",
    "ip",
    "op_category",
)


@dataclass(frozen=True)
class Transition:
    """Transition probability to the next operation."""

    op: str
    prob: float


@dataclass(frozen=True)
class OperationSpec:
    """Definition of a single operation (categorical transitions and Δt)."""

    method: str
    path: str
    referer: str
    user_agent: str
    ip: str
    op_category: str
    transitions: Sequence[Transition]
    cdf_points: Sequence[tuple[float, float]]


@dataclass(frozen=True)
class TimeAnomalySpec:
    """Configuration for time-based anomaly injection."""

    mode: str
    probability: float
    op: str | None
    scale: float | None
    delta: float | None


@dataclass(frozen=True)
class OrderAnomalySpec:
    """Configuration for order-violation anomalies."""

    probability: float
    op: str | None


@dataclass(frozen=True)
class UserSpec:
    """Configuration for a synthetic user/session timeline."""

    uid: str
    session_id: str
    initial_op: str
    steps: int


@dataclass(frozen=True)
class ScenarioSpec:
    """Resolved scenario specification for log generation."""

    algo_version: str
    t0: datetime
    users: Sequence[UserSpec]
    operations: dict[str, OperationSpec]
    time_anomalies: Sequence[TimeAnomalySpec]
    order_anomalies: Sequence[OrderAnomalySpec]


@dataclass
class _GeneratedEvent:
    """Intermediate representation for a generated record."""

    timestamp: datetime
    uid: str
    session_id: str
    op_name: str
    op_spec: OperationSpec
    index: int
    user_step: int
    dt_seconds: float


class ScenarioSpecError(ValueError):
    """Raised when the scenario specification violates the contract."""


def load_spec(path: Path) -> ScenarioSpec:
    """Load and validate a scenario specification from JSON."""

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - sanity fallback
        raise ScenarioSpecError(f"scenario_spec.json is not valid JSON: {exc}") from exc

    if "algo_version" not in data:
        raise ScenarioSpecError("scenario spec must provide 'algo_version'")
    algo_version = str(data["algo_version"])

    t0_raw = data.get("t0", "1970-01-01T00:00:00Z")
    t0 = _parse_utc_timestamp(t0_raw)

    users = [_parse_user_spec(item) for item in data.get("users", [])]
    if not users:
        raise ScenarioSpecError("scenario spec must list at least one user")

    operations = _parse_operations(data.get("ops", {}))
    time_anomalies, order_anomalies = _parse_anomalies(data.get("anoms", []))
    return ScenarioSpec(
        algo_version=algo_version,
        t0=t0,
        users=users,
        operations=operations,
        time_anomalies=time_anomalies,
        order_anomalies=order_anomalies,
    )


def _parse_utc_timestamp(raw: str) -> datetime:
    if not isinstance(raw, str):
        raise ScenarioSpecError("timestamp values must be ISO8601 strings")
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ScenarioSpecError(f"invalid ISO timestamp: {raw}") from exc

    if timestamp.tzinfo is None:
        raise ScenarioSpecError("timestamps must include UTC offset")
    return timestamp.astimezone(timezone.utc)


def _parse_user_spec(raw: dict) -> UserSpec:
    required = {"uid", "session_id", "initial_op", "steps"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ScenarioSpecError(f"user spec missing keys: {missing}")

    steps = int(raw["steps"])
    if steps <= 0:
        raise ScenarioSpecError("user steps must be positive")

    return UserSpec(
        uid=str(raw["uid"]),
        session_id=str(raw["session_id"]),
        initial_op=str(raw["initial_op"]),
        steps=steps,
    )


def _parse_operations(raw_ops: dict) -> dict[str, OperationSpec]:
    if not raw_ops:
        raise ScenarioSpecError("scenario spec must define 'ops'")

    operations: dict[str, OperationSpec] = {}
    for op_name, op_data in raw_ops.items():
        transitions = _parse_transitions(op_name, op_data.get("transitions", []))
        cdf_points = _parse_cdf(op_name, op_data.get("dt_distribution", {}))
        operations[op_name] = OperationSpec(
            method=str(op_data.get("method", "")),
            path=str(op_data.get("path", "")),
            referer=str(op_data.get("referer", "")),
            user_agent=str(op_data.get("user_agent", "")),
            ip=str(op_data.get("ip", "")),
            op_category=str(op_data.get("op_category", "")),
            transitions=transitions,
            cdf_points=cdf_points,
        )
    return operations


def _parse_anomalies(
    raw_anoms: Iterable[dict],
) -> tuple[list[TimeAnomalySpec], list[OrderAnomalySpec]]:
    time_anomalies: list[TimeAnomalySpec] = []
    order_anomalies: list[OrderAnomalySpec] = []

    for entry in raw_anoms or []:
        if not isinstance(entry, dict):
            raise ScenarioSpecError("anomaly definitions must be objects")

        anomaly_type = entry.get("type")
        if anomaly_type == "time":
            mode = str(entry.get("mode", "")).lower()
            if mode not in {"local", "propagate"}:
                raise ScenarioSpecError(
                    "time anomaly must set mode to 'local' or 'propagate'"
                )

            probability = float(entry.get("p", 0.0))
            if probability < 0 or probability > 1:
                raise ScenarioSpecError(
                    "time anomaly probability must be within [0, 1]"
                )

            scale = entry.get("scale")
            delta = entry.get("delta")
            if scale is None and delta is None:
                raise ScenarioSpecError("time anomaly must define 'scale' or 'delta'")

            scale_value = float(scale) if scale is not None else None
            delta_value = float(delta) if delta is not None else None
            op = entry.get("op")
            op_value = str(op) if op is not None else None

            time_anomalies.append(
                TimeAnomalySpec(
                    mode=mode,
                    probability=probability,
                    op=op_value,
                    scale=scale_value,
                    delta=delta_value,
                )
            )
        elif anomaly_type == "order":
            probability = float(entry.get("p", 0.0))
            if probability < 0 or probability > 1:
                raise ScenarioSpecError(
                    "order anomaly probability must be within [0, 1]"
                )

            op = entry.get("op")
            op_value = str(op) if op is not None else None
            order_anomalies.append(
                OrderAnomalySpec(probability=probability, op=op_value)
            )
        else:
            raise ScenarioSpecError(f"unsupported anomaly type: {anomaly_type}")

    return time_anomalies, order_anomalies


def _parse_transitions(
    op_name: str, raw_transitions: Iterable[dict]
) -> list[Transition]:
    transitions: list[Transition] = []
    total_prob = 0.0
    for item in raw_transitions:
        if "op" not in item or "prob" not in item:
            raise ScenarioSpecError(
                f"transition for '{op_name}' must provide 'op' and 'prob'"
            )
        prob = float(item["prob"])
        if prob < 0:
            raise ScenarioSpecError(
                f"transition probability must be non-negative for '{op_name}'"
            )
        total_prob += prob
        transitions.append(Transition(op=str(item["op"]), prob=prob))

    if not transitions:
        raise ScenarioSpecError(f"operation '{op_name}' must define transitions")

    if total_prob <= 0:
        raise ScenarioSpecError(
            f"transition probabilities must sum to >0 for '{op_name}'"
        )

    return transitions


def _parse_cdf(op_name: str, raw: dict) -> list[tuple[float, float]]:
    if raw.get("type") != "piecewise":
        raise ScenarioSpecError(
            f"operation '{op_name}' must use 'piecewise' dt_distribution"
        )

    cdf_points_raw = raw.get("cdf", [])
    if not cdf_points_raw:
        raise ScenarioSpecError(
            f"operation '{op_name}' must provide dt_distribution.cdf"
        )

    cdf_points: list[tuple[float, float]] = []
    for entry in cdf_points_raw:
        if "p" not in entry or "seconds" not in entry:
            raise ScenarioSpecError(
                f"cdf entry for '{op_name}' must include 'p' and 'seconds'"
            )
        probability = float(entry["p"])
        seconds = float(entry["seconds"])
        if probability < 0 or probability > 1:
            raise ScenarioSpecError(
                f"cdf probability must be within [0, 1] for '{op_name}'"
            )
        if seconds < 0:
            raise ScenarioSpecError(f"cdf seconds must be non-negative for '{op_name}'")
        cdf_points.append((probability, seconds))

    cdf_points.sort(key=lambda item: item[0])
    if not math.isclose(cdf_points[0][0], 0.0, abs_tol=1e-9):
        raise ScenarioSpecError(f"cdf for '{op_name}' must start at p=0")
    if not math.isclose(cdf_points[-1][0], 1.0, abs_tol=1e-9):
        raise ScenarioSpecError(f"cdf for '{op_name}' must end at p=1")

    return cdf_points


def generate_normal_records(spec: ScenarioSpec, seed: int) -> list[dict[str, str]]:
    """Generate contract-compliant records for the normal log."""

    events = _generate_events(spec, seed)
    events.sort(key=lambda event: (event.timestamp, event.index))
    return [_record_from_event(event, event.timestamp) for event in events]


def _generate_events(spec: ScenarioSpec, seed: int) -> list[_GeneratedEvent]:
    rng = random.Random(seed)
    events: list[_GeneratedEvent] = []
    record_index = 0
    for user in spec.users:
        current_time = spec.t0
        current_op = user.initial_op
        user_step = 0
        for _ in range(user.steps):
            op_spec = _resolve_operation(spec, current_op)
            dt_seconds = _sample_dt_seconds(rng, op_spec.cdf_points)
            next_op = _sample_next_operation(rng, op_spec.transitions)
            events.append(
                _GeneratedEvent(
                    timestamp=current_time,
                    uid=user.uid,
                    session_id=user.session_id,
                    op_name=current_op,
                    op_spec=op_spec,
                    index=record_index,
                    user_step=user_step,
                    dt_seconds=dt_seconds,
                )
            )
            record_index += 1
            user_step += 1
            current_time = current_time + timedelta(seconds=dt_seconds)
            current_op = next_op
    return events


def write_contract_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write rows in the canonical contract CSV order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(CONTRACT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_run_meta(path: Path, spec: ScenarioSpec, seed: int, spec_sha256: str) -> None:
    """Persist the deterministic run metadata required by the contract."""

    payload = {
        "seed": seed,
        "algo_version": spec.algo_version,
        "spec_sha256": spec_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def generate_anom_records(
    spec: ScenarioSpec, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Generate anomalous records and corresponding audit entries."""

    events = _generate_events(spec, seed)
    events_by_user = _group_events_by_user(events)
    audit_entries: list[dict[str, Any]] = []

    mutated_times = _apply_time_anomalies(events_by_user, spec, seed, audit_entries)
    _apply_order_anomalies(events_by_user, spec, seed, audit_entries)

    events.sort(
        key=lambda event: (mutated_times.get(event.index, event.timestamp), event.index)
    )
    records = [
        _record_from_event(event, mutated_times.get(event.index, event.timestamp))
        for event in events
    ]
    return records, audit_entries


def write_audit_log(path: Path, entries: Sequence[dict[str, Any]]) -> None:
    """Persist the audit trail as newline-delimited JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry, sort_keys=True))
            fp.write("\n")


def _sample_dt_seconds(
    rng: random.Random, cdf_points: Sequence[tuple[float, float]]
) -> float:
    u = rng.random()
    previous_p, previous_seconds = cdf_points[0]
    if u <= previous_p:
        return previous_seconds

    for probability, seconds in cdf_points[1:]:
        if u <= probability:
            if probability == previous_p:
                return seconds
            ratio = (u - previous_p) / (probability - previous_p)
            return previous_seconds + ratio * (seconds - previous_seconds)
        previous_p, previous_seconds = probability, seconds
    return cdf_points[-1][1]


def _sample_next_operation(
    rng: random.Random, transitions: Sequence[Transition]
) -> str:
    total_prob = sum(item.prob for item in transitions)
    threshold = rng.random() * total_prob
    cumulative = 0.0
    for item in transitions:
        cumulative += item.prob
        if threshold <= cumulative:
            return item.op
    return transitions[-1].op


def _group_events_by_user(
    events: Sequence[_GeneratedEvent],
) -> dict[tuple[str, str], list[_GeneratedEvent]]:
    groups: dict[tuple[str, str], list[_GeneratedEvent]] = {}
    for event in events:
        key = (event.uid, event.session_id)
        groups.setdefault(key, []).append(event)
    for user_events in groups.values():
        user_events.sort(key=lambda item: item.user_step)
    return groups


def _apply_time_anomalies(
    events_by_user: dict[tuple[str, str], list[_GeneratedEvent]],
    spec: ScenarioSpec,
    seed: int,
    audit_entries: list[dict[str, Any]],
) -> dict[int, datetime]:
    mutated_times: dict[int, datetime] = {}
    rng = random.Random(seed + 1)

    for user_events in events_by_user.values():
        if not user_events:
            continue

        current_time = user_events[0].timestamp
        propagate_multiplier = 1.0
        propagate_delta = 0.0

        for event in user_events:
            mutated_times[event.index] = current_time
            mutated_dt = event.dt_seconds * propagate_multiplier + propagate_delta

            for time_spec in spec.time_anomalies:
                if time_spec.op is not None and time_spec.op != event.op_name:
                    continue

                if rng.random() <= time_spec.probability:
                    before = mutated_dt
                    mutated_dt = _apply_time_adjustment(mutated_dt, time_spec)
                    if time_spec.mode == "propagate":
                        if time_spec.scale is not None:
                            propagate_multiplier *= time_spec.scale
                        if time_spec.delta is not None:
                            propagate_delta += time_spec.delta

                    audit_entry: dict[str, Any] = {
                        "type": "time",
                        "mode": time_spec.mode,
                        "seed": seed,
                        "record_index": event.index,
                        "op": event.op_name,
                        "seconds_before": before,
                        "seconds_after": max(mutated_dt, 0.0),
                    }
                    if time_spec.scale is not None:
                        audit_entry["scale"] = time_spec.scale
                    if time_spec.delta is not None:
                        audit_entry["delta"] = time_spec.delta
                    audit_entries.append(audit_entry)

            mutated_dt = max(mutated_dt, 0.0)
            current_time = current_time + timedelta(seconds=mutated_dt)

    return mutated_times


def _apply_order_anomalies(
    events_by_user: dict[tuple[str, str], list[_GeneratedEvent]],
    spec: ScenarioSpec,
    seed: int,
    audit_entries: list[dict[str, Any]],
) -> None:
    if not spec.order_anomalies:
        return

    rng = random.Random(seed + 2)
    for order_spec in spec.order_anomalies:
        for user_events in events_by_user.values():
            idx = 0
            while idx < len(user_events) - 1:
                event = user_events[idx]
                next_event = user_events[idx + 1]
                if order_spec.op is not None and event.op_name != order_spec.op:
                    idx += 1
                    continue

                if rng.random() <= order_spec.probability:
                    before_op = event.op_name
                    swapped_op = next_event.op_name
                    audit_entries.append(
                        {
                            "type": "order",
                            "seed": seed,
                            "record_index": event.index,
                            "swap_with": next_event.index,
                            "op_before": before_op,
                            "op_after": swapped_op,
                        }
                    )
                    event.op_name, next_event.op_name = (
                        next_event.op_name,
                        event.op_name,
                    )
                    event.op_spec, next_event.op_spec = (
                        next_event.op_spec,
                        event.op_spec,
                    )
                    idx += 2
                else:
                    idx += 1


def _apply_time_adjustment(value: float, spec: TimeAnomalySpec) -> float:
    result = value
    if spec.scale is not None:
        result *= spec.scale
    if spec.delta is not None:
        result += spec.delta
    return result


def _resolve_operation(spec: ScenarioSpec, op_name: str) -> OperationSpec:
    if op_name not in spec.operations:
        raise ScenarioSpecError(f"operation '{op_name}' not found in scenario spec")
    return spec.operations[op_name]


def _format_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _record_from_event(event: _GeneratedEvent, timestamp: datetime) -> dict[str, str]:
    return {
        "timestamp_utc": _format_timestamp(timestamp),
        "uid": event.uid,
        "session_id": event.session_id,
        "method": event.op_spec.method,
        "path": event.op_spec.path,
        "referer": event.op_spec.referer,
        "user_agent": event.op_spec.user_agent,
        "ip": event.op_spec.ip,
        "op_category": event.op_spec.op_category,
    }
