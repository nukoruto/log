"""Core generation utilities for normal log synthesis."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

CONTRACT_COLUMNS = [
    "timestamp_utc",
    "uid",
    "session_id",
    "method",
    "path",
    "referer",
    "user_agent",
    "ip",
    "op_category",
]


@dataclass(frozen=True)
class ScenarioSpec:
    """Representation of the scenario specification required for generation."""

    length: int
    t0: datetime
    user_count: int
    uid_hmac_key: bytes
    pi: Mapping[str, float]
    transitions: Mapping[str, Mapping[str, float]]
    mu: Mapping[str, float]
    sigma: Mapping[str, float]
    min_seconds: float
    catalog: Mapping[str, Mapping[str, str]]
    algo_version: str


class SpecValidationError(ValueError):
    """Raised when the scenario specification violates the contract."""


def load_spec(path: Path) -> ScenarioSpec:
    """Load and validate the scenario specification."""

    if not path.exists():
        raise SpecValidationError(f"scenario spec not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    try:
        length = int(raw["length"])
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise SpecValidationError("`length` must be provided as integer") from exc
    if length <= 0:
        raise SpecValidationError("`length` must be greater than zero")

    t0_str = raw.get("t0", "1970-01-01T00:00:00Z")
    t0 = _parse_utc_timestamp(t0_str)

    users_section = raw.get("users")
    if isinstance(users_section, int):
        user_count = users_section
    elif isinstance(users_section, Mapping):
        try:
            user_count = int(users_section["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecValidationError("`users.count` must be provided") from exc
    else:
        raise SpecValidationError("`users` must be an integer or mapping with `count`")
    if user_count <= 0:
        raise SpecValidationError("`users.count` must be positive")

    uid_hmac_key = raw.get("uid_hmac_key")
    if not isinstance(uid_hmac_key, str) or not uid_hmac_key:
        raise SpecValidationError("`uid_hmac_key` must be a non-empty string")

    pi = raw.get("pi")
    if not isinstance(pi, Mapping) or not pi:
        raise SpecValidationError("`pi` must be a non-empty mapping of op categories")
    pi = {str(k): float(v) for k, v in pi.items()}
    _validate_probabilities(pi.values(), "pi")

    transitions = raw.get("A")
    if not isinstance(transitions, Mapping) or not transitions:
        raise SpecValidationError("`A` must be a non-empty mapping of transitions")
    norm_transitions: Dict[str, Dict[str, float]] = {}
    for src, dst_map in transitions.items():
        if not isinstance(dst_map, Mapping) or not dst_map:
            raise SpecValidationError(f"transition row for '{src}' must be a mapping")
        row = {str(k): float(v) for k, v in dst_map.items()}
        _validate_probabilities(row.values(), f"A[{src}]")
        norm_transitions[str(src)] = row

    dt_section = raw.get("dt")
    if not isinstance(dt_section, Mapping):
        raise SpecValidationError("`dt` section is required")
    lognorm = dt_section.get("lognorm")
    if not isinstance(lognorm, Mapping):
        raise SpecValidationError("`dt.lognorm` section is required")
    mu_section = lognorm.get("mu")
    sigma_section = lognorm.get("sigma")
    if not isinstance(mu_section, Mapping) or not isinstance(sigma_section, Mapping):
        raise SpecValidationError("`dt.lognorm.mu` and `.sigma` must be provided")
    mu = {str(k): float(v) for k, v in mu_section.items()}
    sigma = {str(k): float(v) for k, v in sigma_section.items()}
    for key in pi:
        if key not in mu or key not in sigma:
            raise SpecValidationError(f"missing log-normal parameters for op '{key}'")
        if sigma[key] < 0:
            raise SpecValidationError(
                f"log-normal sigma for '{key}' must be non-negative"
            )
    min_seconds = float(dt_section.get("min_seconds", 1e-6))
    if min_seconds <= 0:
        raise SpecValidationError("`dt.min_seconds` must be positive")

    catalog = raw.get("catalog")
    if not isinstance(catalog, Mapping) or not catalog:
        raise SpecValidationError(
            "`catalog` must map each op_category to HTTP metadata"
        )
    normalized_catalog: Dict[str, Dict[str, str]] = {}
    for op, entry in catalog.items():
        if not isinstance(entry, Mapping):
            raise SpecValidationError(f"catalog entry for '{op}' must be a mapping")
        normalized_catalog[op] = {
            "method": str(entry.get("method", "")),
            "path": str(entry.get("path", "")),
            "referer": str(entry.get("referer", "")),
            "user_agent": str(entry.get("user_agent", "")),
            "ip": str(entry.get("ip", "")),
        }
        for field in ("method", "path", "user_agent", "ip"):
            if not normalized_catalog[op][field]:
                raise SpecValidationError(
                    f"catalog entry for '{op}' must include non-empty '{field}'"
                )
    for key in pi:
        if key not in normalized_catalog:
            raise SpecValidationError(f"catalog missing entry for op '{key}'")

    algo_version = str(raw.get("algo_version", "0.0.0"))

    return ScenarioSpec(
        length=length,
        t0=t0,
        user_count=user_count,
        uid_hmac_key=uid_hmac_key.encode("utf-8"),
        pi=pi,
        transitions=norm_transitions,
        mu=mu,
        sigma=sigma,
        min_seconds=min_seconds,
        catalog=normalized_catalog,
        algo_version=algo_version,
    )


def generate_normal_records(spec: ScenarioSpec, seed: int) -> List[Dict[str, str]]:
    """Generate normal log records based on the provided scenario specification."""

    rng = random.Random(seed)
    categories = list(spec.pi.keys())
    pi_weights = _normalise_weights(spec.pi.values())

    current_op = _weighted_choice(rng, categories, pi_weights)
    current_time = spec.t0

    user_indices = list(range(spec.user_count))
    user_ids = [_anonymise(spec.uid_hmac_key, f"user-{idx}") for idx in user_indices]
    session_ids = [
        _anonymise(spec.uid_hmac_key, f"session-{idx}") for idx in user_indices
    ]

    records: List[Dict[str, str]] = []
    for _ in range(spec.length):
        user_index = rng.choice(user_indices)
        record = _build_record(
            timestamp=current_time,
            uid=user_ids[user_index],
            session_id=session_ids[user_index],
            op=current_op,
            catalog=spec.catalog,
        )
        records.append(record)

        dt_seconds = max(
            spec.min_seconds,
            _sample_lognormal(rng, spec.mu[current_op], spec.sigma[current_op]),
        )
        current_time = current_time + timedelta(seconds=dt_seconds)
        current_op = _sample_next_operation(rng, current_op, spec.transitions)

    return records


def write_contract_csv(records: Sequence[Mapping[str, str]], path: Path) -> None:
    """Write generated records to a CSV file following the contract order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRACT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def format_utc(dt: datetime) -> str:
    """Format a timezone-aware UTC datetime into the contract string representation."""

    return (
        dt.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise SpecValidationError("timestamps must include UTC offset")
    return dt.astimezone(timezone.utc)


def _validate_probabilities(probabilities: Iterable[float], field: str) -> None:
    total = 0.0
    for p in probabilities:
        if p < 0:
            raise SpecValidationError(f"probabilities in {field} must be non-negative")
        total += p
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise SpecValidationError(f"probabilities in {field} must sum to 1.0")


def _normalise_weights(weights: Iterable[float]) -> List[float]:
    data = list(weights)
    total = sum(data)
    if total == 0:
        raise SpecValidationError("probabilities cannot sum to zero")
    return [w / total for w in data]


def _weighted_choice(
    rng: random.Random, options: Sequence[str], weights: Sequence[float]
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for option, weight in zip(options, weights):
        cumulative += weight
        if threshold <= cumulative:
            return option
    return options[-1]


def _sample_next_operation(
    rng: random.Random, current: str, transitions: Mapping[str, Mapping[str, float]]
) -> str:
    if current not in transitions:
        raise SpecValidationError(f"missing transition probabilities for '{current}'")
    next_ops = list(transitions[current].keys())
    weights = _normalise_weights(transitions[current].values())
    return _weighted_choice(rng, next_ops, weights)


def _sample_lognormal(rng: random.Random, mu: float, sigma: float) -> float:
    if sigma == 0:
        return math.exp(mu)
    return rng.lognormvariate(mu, sigma)


def _anonymise(key: bytes, value: str) -> str:
    digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def _build_record(
    *,
    timestamp: datetime,
    uid: str,
    session_id: str,
    op: str,
    catalog: Mapping[str, Mapping[str, str]],
) -> Dict[str, str]:
    if op not in catalog:
        raise SpecValidationError(f"catalog does not provide metadata for '{op}'")
    metadata = catalog[op]
    return {
        "timestamp_utc": format_utc(timestamp),
        "uid": uid,
        "session_id": session_id,
        "method": metadata["method"],
        "path": metadata["path"],
        "referer": metadata.get("referer", ""),
        "user_agent": metadata["user_agent"],
        "ip": metadata["ip"],
        "op_category": op,
    }


def compute_spec_sha256(path: Path) -> str:
    """Compute the SHA256 hash of the specification file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
