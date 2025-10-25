"""Utilities for injecting anomalies during log generation."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .generate import (
        ScenarioSpec,
        TimeAnomalySpec,
        TokenReplayAnomalySpec,
        UnauthAnomalySpec,
        _GeneratedEvent,
    )


_ORDER_PATCHED = False


def apply_time_anomalies(
    events_by_user: Mapping[tuple[str, str], Sequence["_GeneratedEvent"]],
    time_specs: Sequence["TimeAnomalySpec"],
    seed: int,
    audit_entries: list[dict[str, Any]],
) -> dict[int, datetime]:
    """Apply time anomalies to the generated events.

    Args:
        events_by_user: Events grouped by (uid, session_id).
        time_specs: Time anomaly specifications resolved from the scenario.
        seed: Base seed for deterministic sampling.
        audit_entries: List that will be appended with audit dictionaries.

    Returns:
        Mapping from record index to the mutated timestamp.
    """

    _ensure_order_patch()

    mutated_times: dict[int, datetime] = {}
    if not time_specs:
        for user_events in events_by_user.values():
            for event in user_events:
                mutated_times[event.index] = event.timestamp
        return mutated_times

    rng = random.Random(seed + 1)

    for user_events in events_by_user.values():
        if not user_events:
            continue

        current_time = user_events[0].timestamp
        propagate_multiplier = 1.0
        propagate_delta = 0.0

        for event in user_events:
            mutated_times[event.index] = current_time
            base_dt = event.dt_seconds * propagate_multiplier + propagate_delta
            mutated_dt = base_dt

            for time_spec in time_specs:
                if time_spec.op is not None and time_spec.op != event.op_name:
                    continue

                draw = rng.random()
                if draw > time_spec.probability:
                    continue

                before = mutated_dt
                mutated_dt = _apply_time_adjustment(mutated_dt, time_spec)

                if time_spec.mode == "propagate":
                    if time_spec.scale is not None:
                        propagate_multiplier *= time_spec.scale
                    if time_spec.delta is not None:
                        propagate_delta += time_spec.delta

                audit_entries.append(
                    _build_time_audit_entry(
                        seed=seed,
                        event=event,
                        spec=time_spec,
                        seconds_before=before,
                        seconds_after=max(mutated_dt, 0.0),
                    )
                )

            mutated_dt = max(mutated_dt, 0.0)
            current_time = current_time + timedelta(seconds=mutated_dt)

    return mutated_times


def _apply_time_adjustment(value: float, spec: "TimeAnomalySpec") -> float:
    result = value
    if spec.scale is not None:
        result *= spec.scale
    if spec.delta is not None:
        result += spec.delta
    return result


def _build_time_audit_entry(
    *,
    seed: int,
    event: "_GeneratedEvent",
    spec: "TimeAnomalySpec",
    seconds_before: float,
    seconds_after: float,
) -> dict[str, Any]:
    params: dict[str, float] = {}
    param_parts: list[str] = []
    if spec.scale is not None:
        params["scale"] = spec.scale
        param_parts.append(f"scale={spec.scale}")
    if spec.delta is not None:
        params["delta"] = spec.delta
        param_parts.append(f"delta={spec.delta}")
    param_text = ", ".join(param_parts) if param_parts else "no parameters"

    reason = (
        f"Applied {spec.mode} time anomaly to record {event.index} with {param_text}; "
        f"seconds {seconds_before:.3f}->{seconds_after:.3f}."
    )
    if spec.mode == "propagate":
        reason += " Future events re-integrated per Eq. C.2."
    else:
        reason += " Effect limited to current record."

    entry: dict[str, Any] = {
        "type": "time",
        "mode": spec.mode,
        "seed": seed,
        "record_index": event.index,
        "op": event.op_name,
        "seconds_before": seconds_before,
        "seconds_after": seconds_after,
        "params": params,
        "reason": reason,
    }
    if spec.scale is not None:
        entry["scale"] = spec.scale
    if spec.delta is not None:
        entry["delta"] = spec.delta
    return entry


def _ensure_order_patch() -> None:
    """Ensure the order anomaly handler in ``generate`` uses the patched logic."""

    global _ORDER_PATCHED
    if _ORDER_PATCHED:
        return

    module = sys.modules.get("log_generator.generate")
    if module is None:
        return

    module._apply_order_anomalies = _apply_order_anomalies  # type: ignore[attr-defined]
    _ORDER_PATCHED = True


def _apply_order_anomalies(
    events_by_user: Mapping[tuple[str, str], Sequence["_GeneratedEvent"]],
    spec: "ScenarioSpec",
    seed: int,
    audit_entries: list[dict[str, Any]],
) -> None:
    order_specs = getattr(spec, "order_anomalies", [])
    if not order_specs:
        return

    rng = random.Random(seed + 2)
    for order_spec in order_specs:
        for user_events in events_by_user.values():
            if len(user_events) < 2:
                continue

            idx = 0
            while idx < len(user_events) - 1:
                event = user_events[idx]
                next_event = user_events[idx + 1]

                if order_spec.op is not None and event.op_name != order_spec.op:
                    idx += 1
                    continue

                if rng.random() > order_spec.probability:
                    idx += 1
                    continue

                original_event_op = event.op_name
                original_next_op = next_event.op_name

                event.op_name, next_event.op_name = (
                    next_event.op_name,
                    event.op_name,
                )
                event.op_spec, next_event.op_spec = (
                    next_event.op_spec,
                    event.op_spec,
                )

                audit_entries.append(
                    _build_order_audit_entry(
                        seed=seed,
                        first=event,
                        second=next_event,
                        op_before=original_event_op,
                        swap_before=original_next_op,
                    )
                )
                _update_time_audit_entries(
                    audit_entries,
                    event_index=event.index,
                    event_op=event.op_name,
                    swap_index=next_event.index,
                    swap_op=next_event.op_name,
                )

                idx += 2


def _build_order_audit_entry(
    *,
    seed: int,
    first: "_GeneratedEvent",
    second: "_GeneratedEvent",
    op_before: str,
    swap_before: str,
) -> dict[str, Any]:
    constraint = f"{op_before}->{swap_before}"
    reason = (
        "Swapped records "
        f"{first.index}({op_before}) and {second.index}({swap_before}) to violate "
        f"order constraint ℛ({constraint})."
    )

    return {
        "type": "order",
        "seed": seed,
        "record_index": first.index,
        "swap_with": second.index,
        "op_before": op_before,
        "op_after": first.op_name,
        "swap_op_before": swap_before,
        "swap_op_after": second.op_name,
        "constraint": constraint,
        "reason": reason,
    }


def _update_time_audit_entries(
    audit_entries: list[dict[str, Any]],
    *,
    event_index: int,
    event_op: str,
    swap_index: int,
    swap_op: str,
) -> None:
    for entry in audit_entries:
        if entry.get("type") != "time":
            continue
        record_index = entry.get("record_index")
        if record_index == event_index:
            entry["op"] = event_op
        elif record_index == swap_index:
            entry["op"] = swap_op


def apply_unauth_anomalies(
    *,
    events_by_user: Mapping[tuple[str, str], Sequence["_GeneratedEvent"]],
    unauth_specs: Sequence["UnauthAnomalySpec"],
    seed: int,
    audit_entries: list[dict[str, Any]],
) -> None:
    """Inject unauthenticated access anomalies by mutating session identifiers."""

    if not unauth_specs:
        return

    rng = random.Random(seed + 3)
    for user_key in sorted(events_by_user):
        for event in events_by_user[user_key]:
            if getattr(event, "_unauth_injected", False):
                continue
            if not _requires_authentication(event):
                continue

            for unauth_spec in unauth_specs:
                if unauth_spec.op is not None and unauth_spec.op != event.op_name:
                    continue
                if rng.random() > unauth_spec.probability:
                    continue

                original_session = event.session_id
                mutated_session = _mutate_session_for_unauth(
                    original_session, seed, event.index
                )
                event.session_id = mutated_session
                setattr(event, "_unauth_injected", True)
                audit_entries.append(
                    _build_unauth_audit_entry(
                        seed=seed,
                        event=event,
                        original_session=original_session,
                        mutated_session=mutated_session,
                    )
                )
                break


def apply_token_replay_anomalies(
    *,
    events_by_user: Mapping[tuple[str, str], Sequence["_GeneratedEvent"]],
    token_specs: Sequence["TokenReplayAnomalySpec"],
    seed: int,
    audit_entries: list[dict[str, Any]],
) -> None:
    """Inject token replay anomalies by reusing session identifiers across users."""

    if not token_specs:
        return

    session_owners = _collect_session_owners(events_by_user)
    if len(session_owners) < 2:
        return

    ordered_sessions = sorted(session_owners.items())
    rng = random.Random(seed + 4)
    for user_key in sorted(events_by_user):
        for event in events_by_user[user_key]:
            if getattr(event, "_unauth_injected", False):
                continue
            if getattr(event, "_token_replay_injected", False):
                continue

            for token_spec in token_specs:
                if token_spec.op is not None and token_spec.op != event.op_name:
                    continue
                if rng.random() > token_spec.probability:
                    continue

                candidates = [
                    candidate
                    for candidate in ordered_sessions
                    if candidate[0] != event.session_id and candidate[1] != event.uid
                ]
                if not candidates:
                    continue

                replay_session, owner_uid = rng.choice(candidates)
                original_session = event.session_id
                event.session_id = replay_session
                setattr(event, "_token_replay_injected", True)
                audit_entries.append(
                    _build_token_replay_audit_entry(
                        seed=seed,
                        event=event,
                        original_session=original_session,
                        replayed_session=replay_session,
                        owner_uid=owner_uid,
                    )
                )
                break


def _requires_authentication(event: "_GeneratedEvent") -> bool:
    category = event.op_spec.op_category.strip().upper()
    return category in {"UPDATE", "DELETE"}


def _mutate_session_for_unauth(original: str, seed: int, index: int) -> str:
    base = original if original else f"sess-{seed}"
    return f"{base}-unauth-{seed}-{index}"


def _build_unauth_audit_entry(
    *,
    seed: int,
    event: "_GeneratedEvent",
    original_session: str,
    mutated_session: str,
) -> dict[str, Any]:
    reason = (
        f"Forced AuthOK violation on record {event.index} ({event.op_name}) by "
        f"replacing session '{original_session}' with '{mutated_session}'."
    )
    reason += f" AuthOK predicate failed for category {event.op_spec.op_category}."

    return {
        "type": "unauth",
        "seed": seed,
        "record_index": event.index,
        "op": event.op_name,
        "uid": event.uid,
        "op_category": event.op_spec.op_category,
        "session_original": original_session,
        "session_mutated": mutated_session,
        "auth_required": True,
        "reason": reason,
    }


def _collect_session_owners(
    events_by_user: Mapping[tuple[str, str], Sequence["_GeneratedEvent"]],
) -> dict[str, str]:
    owners: dict[str, str] = {}
    for uid, session_id in events_by_user.keys():
        owners.setdefault(session_id, uid)
    return owners


def _build_token_replay_audit_entry(
    *,
    seed: int,
    event: "_GeneratedEvent",
    original_session: str,
    replayed_session: str,
    owner_uid: str,
) -> dict[str, Any]:
    reason = (
        f"Replayed session '{replayed_session}' originally issued to uid "
        f"'{owner_uid}' on record {event.index}; uid '{event.uid}' now "
        "presents the token, violating uid×session_id integrity."
    )

    return {
        "type": "token_replay",
        "seed": seed,
        "record_index": event.index,
        "op": event.op_name,
        "uid": event.uid,
        "op_category": event.op_spec.op_category,
        "session_original": original_session,
        "session_replayed": replayed_session,
        "token_owner_uid": owner_uid,
        "reason": reason,
    }
