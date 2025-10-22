"""Statistical estimators for the scenario-design fit command."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_EPS: float = 1e-3
ALGO_VERSION: str = "B1.0.0"
KNOWN_OPERATIONS: frozenset[str] = frozenset(
    {
        "AUTH",
        "READ",
        "WRITE",
        "UPDATE",
        "DELETE",
        "CREATE",
        "TOKEN",
        "ADMIN",
        "EXPORT",
        "IMPORT",
        "SEARCH",
    }
)


@dataclass(slots=True)
class SessionEvent:
    """A single operation event grouped by session."""

    timestamp: datetime
    op_category: str


@dataclass(slots=True)
class FitStatistics:
    """Container for statistics estimated from deltified logs."""

    ops: list[str]
    pi: dict[str, float]
    A: dict[str, dict[str, float]]
    dt: dict[str, dict[str, float]]
    input_sha256: str
    algo_version: str = ALGO_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert the statistics to a JSON/pickle friendly dict."""

        return {
            "ops": list(self.ops),
            "pi": {k: float(v) for k, v in self.pi.items()},
            "A": {
                row: {col: float(p) for col, p in cols.items()}
                for row, cols in self.A.items()
            },
            "dt": {
                "mu": {k: float(v) for k, v in self.dt["mu"].items()},
                "sigma": {k: float(v) for k, v in self.dt["sigma"].items()},
            },
            "input_sha256": self.input_sha256,
            "algo_version": self.algo_version,
        }


def _parse_timestamp(value: str) -> datetime:
    """Parse ISO8601 timestamps with UTC enforcement."""

    text = value.strip()
    if not text:
        raise ValueError("timestamp_utc must not be empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"timestamp_utc has invalid format: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp_utc must include timezone information (UTC)")
    return parsed.astimezone(timezone.utc)


def _read_rows(
    path: Path,
) -> tuple[dict[str, list[SessionEvent]], dict[str, list[float]], set[str]]:
    required_columns = {
        "timestamp_utc",
        "session_id",
        "op_category",
        "delta_t",
    }
    session_events: dict[str, list[SessionEvent]] = defaultdict(list)
    dt_values: dict[str, list[float]] = defaultdict(list)
    ops_seen: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("deltified.csv must include headers")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            op = row["op_category"].strip()
            if op not in KNOWN_OPERATIONS:
                raise ValueError(f"Encountered unknown operation category: {op}")
            ops_seen.add(op)

            session_id = row["session_id"].strip()
            if not session_id:
                raise ValueError("session_id must not be empty")

            timestamp = _parse_timestamp(row["timestamp_utc"])
            try:
                delta_t = float(row["delta_t"])
            except (
                TypeError,
                ValueError,
            ) as exc:  # pragma: no cover - defensive branch
                raise ValueError(
                    f"delta_t must be numeric, got {row['delta_t']!r}"
                ) from exc
            if delta_t < 0:
                raise ValueError("delta_t must be non-negative")

            session_events[session_id].append(SessionEvent(timestamp, op))
            dt_values[op].append(delta_t)

    if not session_events:
        raise ValueError("No rows found in deltified.csv")

    return session_events, dt_values, ops_seen


def _compute_pi(
    session_events: dict[str, list[SessionEvent]], ops: list[str], alpha: float
) -> dict[str, float]:
    first_counts = {op: 0.0 for op in ops}
    for events in session_events.values():
        events.sort(key=lambda item: item.timestamp)
        first_counts[events[0].op_category] += 1

    total_sessions = len(session_events)
    denom = total_sessions + alpha * len(ops)
    return {op: (first_counts[op] + alpha) / denom for op in ops}


def _compute_transition_matrix(
    session_events: dict[str, list[SessionEvent]], ops: list[str], alpha: float
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {
        op: {dest: 0.0 for dest in ops} for op in ops
    }

    for events in session_events.values():
        events.sort(key=lambda item: item.timestamp)
        for current, nxt in zip(events, events[1:]):
            counts[current.op_category][nxt.op_category] += 1

    transition_matrix: dict[str, dict[str, float]] = {}
    for op in ops:
        outgoing = counts[op]
        total = sum(outgoing.values())
        denom = total + alpha * len(ops)
        transition_matrix[op] = {dest: (outgoing[dest] + alpha) / denom for dest in ops}
    return transition_matrix


def _compute_log_delta_stats(
    dt_values: dict[str, list[float]], ops: list[str]
) -> dict[str, dict[str, float]]:
    mu: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for op in ops:
        values = dt_values.get(op, [])
        if not values:
            raise ValueError(f"No delta_t observations for operation {op}")
        logs = [math.log(value + LOG_EPS) for value in values]
        mu_value = sum(logs) / len(logs)
        variance = sum((value - mu_value) ** 2 for value in logs) / len(logs)
        sigma_value = math.sqrt(variance)
        mu[op] = float(mu_value)
        sigma[op] = float(sigma_value)
    return {"mu": mu, "sigma": sigma}


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_statistics(csv_path: Path | str) -> FitStatistics:
    """Estimate Markov chain and lognormal parameters from deltified CSV."""

    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")

    session_events, dt_values, ops_seen = _read_rows(path)
    ops = sorted(ops_seen)
    alpha = 1.0 / len(ops)

    pi = _compute_pi(session_events, ops, alpha)
    transition_matrix = _compute_transition_matrix(session_events, ops, alpha)
    dt_stats = _compute_log_delta_stats(dt_values, ops)
    input_sha = _sha256_of_file(path)

    return FitStatistics(
        ops=ops,
        pi=pi,
        A=transition_matrix,
        dt=dt_stats,
        input_sha256=input_sha,
    )


__all__ = [
    "estimate_statistics",
    "FitStatistics",
    "LOG_EPS",
    "ALGO_VERSION",
    "KNOWN_OPERATIONS",
]
