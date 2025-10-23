"""Estimation routines for scenario design statistics."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from . import __version__

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "uid",
    "session_id",
    "method",
    "path",
    "referer",
    "user_agent",
    "ip",
    "op_category",
    "delta_t",
]

UNKNOWN_TOKENS = {"__unknown__", "__unk__", "unknown", "unk"}
EPSILON = 1e-9


@dataclass(frozen=True)
class Event:
    """Single record from deltified.csv."""

    timestamp: datetime
    session_id: str
    op_category: str
    delta_t: float


def load_events(path: Path) -> List[Event]:
    """Load deltified events, enforcing required columns and UTC timestamps."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    events: List[Event] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("deltified.csv is missing a header row")
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        for row in reader:
            op_raw = row["op_category"].strip()
            if op_raw.lower() in UNKNOWN_TOKENS:
                raise ValueError("unknown op encountered in op_category column")

            timestamp_raw = row["timestamp_utc"].strip()
            timestamp = _parse_utc_timestamp(timestamp_raw)
            session_id = row["session_id"].strip()
            delta_t = float(row["delta_t"].strip())

            events.append(
                Event(
                    timestamp=timestamp,
                    session_id=session_id,
                    op_category=op_raw,
                    delta_t=delta_t,
                )
            )

    events.sort(key=lambda event: (event.session_id, event.timestamp))
    return events


def _parse_utc_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp_utc must contain timezone information")
    return timestamp.astimezone(timezone.utc)


def compute_categories(events: Sequence[Event]) -> List[str]:
    categories = sorted({event.op_category for event in events})
    if not categories:
        raise ValueError("No events found in deltified.csv")
    return categories


def estimate_initial_distribution(
    events: Sequence[Event], categories: Sequence[str]
) -> Dict[str, float]:
    alpha = 1.0 / len(categories)
    session_first_ops: Dict[str, str] = {}
    for event in events:
        session_first_ops.setdefault(event.session_id, event.op_category)

    counts = Counter(session_first_ops.values())
    total = sum(counts.values())
    denominator = total + alpha * len(categories)
    distribution: Dict[str, float] = {}
    for category in categories:
        distribution[category] = (counts.get(category, 0) + alpha) / denominator
    return distribution


def estimate_transition_matrix(
    events: Sequence[Event], categories: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    alpha = 1.0 / len(categories)
    transitions: Dict[str, Counter[str]] = {
        category: Counter() for category in categories
    }

    previous_by_session: Dict[str, str] = {}
    for event in events:
        previous = previous_by_session.get(event.session_id)
        if previous is not None:
            transitions[previous][event.op_category] += 1
        previous_by_session[event.session_id] = event.op_category

    matrix: Dict[str, Dict[str, float]] = {}
    for source in categories:
        row_counts = transitions[source]
        total = sum(row_counts.values())
        denominator = total + alpha * len(categories)
        matrix[source] = {}
        for target in categories:
            matrix[source][target] = (row_counts.get(target, 0) + alpha) / denominator
    return matrix


def estimate_lognormal_params(
    events: Sequence[Event], categories: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[float]] = {category: [] for category in categories}
    for event in events:
        values[event.op_category].append(math.log(event.delta_t + EPSILON))

    mu: Dict[str, float] = {}
    sigma: Dict[str, float] = {}
    for category in categories:
        series = values[category]
        if not series:
            raise ValueError(f"No delta_t values for category {category}")
        mean = sum(series) / len(series)
        variance = sum((value - mean) ** 2 for value in series) / len(series)
        mu[category] = mean
        sigma[category] = math.sqrt(variance)
    return {"mu": mu, "sigma": sigma}


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed supported random number generators for determinism."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:  # pragma: no cover - optional dependency
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(seed)
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:  # pragma: no cover - depends on CUDA availability
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            pass
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def build_stats(
    events: Sequence[Event],
    input_path: Path,
    *,
    seed: int,
    input_sha256: str | None = None,
) -> Dict[str, object]:
    categories = compute_categories(events)
    pi = estimate_initial_distribution(events, categories)
    transition_matrix = estimate_transition_matrix(events, categories)
    lognorm = estimate_lognormal_params(events, categories)

    sha256 = (
        input_sha256 if input_sha256 is not None else compute_file_sha256(input_path)
    )

    stats: Dict[str, object] = {
        "version": __version__,
        "created_at": events[0].timestamp.isoformat(),
        "seed": seed,
        "input_path": str(input_path),
        "input_sha256": sha256,
        "categories": categories,
        "pi": pi,
        "A": transition_matrix,
        "lognorm": lognorm,
        "n_events": len(events),
        "n_sessions": len({event.session_id for event in events}),
    }
    return stats


def save_stats(stats: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(stats, handle)


def run_fit(
    input_path: Path,
    output_path: Path,
    *,
    seed: int,
    input_sha256: str | None = None,
) -> Dict[str, object]:
    events = load_events(input_path)
    stats = build_stats(events, input_path, seed=seed, input_sha256=input_sha256)
    save_stats(stats, output_path)
    return stats
