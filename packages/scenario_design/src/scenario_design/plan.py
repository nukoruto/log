"""Scenario planning routines for generating normal specifications."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .schema import validate_scenario_spec


class PlanInputError(ValueError):
    """Raised when the provided statistics file is invalid."""


@dataclass(frozen=True)
class PlanResult:
    """Result of generating a scenario specification."""

    spec: Dict[str, Any]
    output_sha256: str


def load_stats(path: Path) -> Dict[str, Any]:
    """Load statistics produced by :mod:`scenario_design.fit`."""

    if not path.exists():
        raise FileNotFoundError(f"Statistics file not found: {path}")

    try:
        data = pickle.loads(path.read_bytes())
    except Exception as exc:  # noqa: BLE001 - propagate with context
        raise PlanInputError(f"Failed to read stats from {path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise PlanInputError("Stats must be a mapping serialized by pickle")

    required_fields = {
        "categories",
        "pi",
        "A",
        "lognorm",
        "n_events",
        "n_sessions",
    }
    missing = required_fields - data.keys()
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise PlanInputError(f"Stats file is missing field(s): {missing_fields}")

    return dict(data)


def _ensure_categories(stats: Mapping[str, Any]) -> list[str]:
    categories = stats["categories"]
    if not isinstance(categories, list) or not categories:
        raise PlanInputError("Stats categories must be a non-empty list")
    if any(not isinstance(category, str) or not category for category in categories):
        raise PlanInputError("Stats categories must contain non-empty strings")
    return list(dict.fromkeys(categories))


def _validate_markov_components(
    stats: Mapping[str, Any], categories: list[str]
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    pi_raw = stats["pi"]
    transitions_raw = stats["A"]
    if not isinstance(pi_raw, Mapping):
        raise PlanInputError("Stats pi must be a mapping of probabilities")
    if not isinstance(transitions_raw, Mapping):
        raise PlanInputError("Stats A must be a mapping of transition rows")

    pi: Dict[str, float] = {}
    for category in categories:
        if category not in pi_raw:
            raise PlanInputError(f"Stats pi missing category: {category}")
        value = pi_raw[category]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PlanInputError(f"Stats pi[{category}] must be numeric")
        pi[category] = float(value)

    if not math.isclose(sum(pi.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise PlanInputError("Stats pi probabilities must sum to 1")

    transitions: Dict[str, Dict[str, float]] = {}
    for source in categories:
        row = transitions_raw.get(source)
        if not isinstance(row, Mapping):
            raise PlanInputError(
                f"Stats A[{source}] must be a mapping of probabilities"
            )
        transitions[source] = {}
        for target in categories:
            if target not in row:
                raise PlanInputError(
                    f"Stats A[{source}] missing transition probability for {target}"
                )
            value = row[target]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PlanInputError(f"Stats A[{source}][{target}] must be numeric")
            transitions[source][target] = float(value)
        if not math.isclose(
            sum(transitions[source].values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise PlanInputError(f"Stats A[{source}] probabilities must sum to 1")

    return pi, transitions


def _validate_lognormal(
    stats: Mapping[str, Any], categories: list[str]
) -> Dict[str, Any]:
    lognorm_raw = stats["lognorm"]
    if not isinstance(lognorm_raw, Mapping):
        raise PlanInputError("Stats lognorm must be a mapping")

    mu_raw = lognorm_raw.get("mu")
    sigma_raw = lognorm_raw.get("sigma")
    if not isinstance(mu_raw, Mapping) or not isinstance(sigma_raw, Mapping):
        raise PlanInputError("Stats lognorm must include mu and sigma mappings")

    mu: Dict[str, float] = {}
    sigma: Dict[str, float] = {}
    for category in categories:
        if category not in mu_raw or category not in sigma_raw:
            raise PlanInputError(
                f"Stats lognorm missing parameters for category: {category}"
            )
        mu_value = mu_raw[category]
        sigma_value = sigma_raw[category]
        if not isinstance(mu_value, (int, float)) or isinstance(mu_value, bool):
            raise PlanInputError(f"Stats lognorm.mu[{category}] must be numeric")
        if not isinstance(sigma_value, (int, float)) or isinstance(sigma_value, bool):
            raise PlanInputError(f"Stats lognorm.sigma[{category}] must be numeric")
        mu[category] = float(mu_value)
        sigma_value_float = float(sigma_value)
        sigma[category] = sigma_value_float if sigma_value_float > 0.0 else 1e-9

    return {"mu": mu, "sigma": sigma}


def build_normal_spec(stats: Mapping[str, Any], *, seed: int) -> Dict[str, Any]:
    """Build a normal scenario specification from statistics."""

    categories = _ensure_categories(stats)
    pi, transitions = _validate_markov_components(stats, categories)
    lognorm = _validate_lognormal(stats, categories)

    length_raw = stats["n_events"]
    users_raw = stats["n_sessions"]
    if (
        not isinstance(length_raw, int)
        or isinstance(length_raw, bool)
        or length_raw <= 0
    ):
        raise PlanInputError("Stats n_events must be a positive integer")
    if not isinstance(users_raw, int) or isinstance(users_raw, bool) or users_raw <= 0:
        raise PlanInputError("Stats n_sessions must be a positive integer")

    ordered_categories = sorted(categories)
    pi_sorted = {category: pi[category] for category in ordered_categories}
    transitions_sorted = {
        source: {target: transitions[source][target] for target in ordered_categories}
        for source in ordered_categories
    }
    lognorm_sorted = {
        "mu": {category: lognorm["mu"][category] for category in ordered_categories},
        "sigma": {
            category: lognorm["sigma"][category] for category in ordered_categories
        },
    }

    spec: Dict[str, Any] = {
        "length": length_raw,
        "users": users_raw,
        "pi": pi_sorted,
        "A": transitions_sorted,
        "dt": {"lognorm": lognorm_sorted},
        "anoms": [],
        "seed": seed,
    }

    validate_scenario_spec(spec)
    return spec


def _dump_spec(spec: Mapping[str, Any]) -> tuple[str, str]:
    json_text = json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True)
    sha256 = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    return json_text, sha256


def save_spec(spec: Mapping[str, Any], output_path: Path) -> str:
    """Save the specification to disk and return its SHA-256 digest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_text, sha256 = _dump_spec(spec)
    output_path.write_text(json_text + "\n", encoding="utf-8")
    return sha256


def run_plan(
    stats_path: Path,
    output_path: Path,
    *,
    seed: int,
) -> PlanResult:
    """Generate a normal scenario specification."""

    stats = load_stats(stats_path)
    spec = build_normal_spec(stats, seed=seed)
    output_sha256 = save_spec(spec, output_path)
    return PlanResult(spec=spec, output_sha256=output_sha256)


__all__ = [
    "PlanInputError",
    "PlanResult",
    "build_normal_spec",
    "load_stats",
    "run_plan",
    "save_spec",
]
