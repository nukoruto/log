"""Utilities for injecting anomalies during log generation."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .generate import TimeAnomalySpec, _GeneratedEvent


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
