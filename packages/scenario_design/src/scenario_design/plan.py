"""Scenario planning routines for generating normal specifications."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .schema import validate_scenario_spec


class PlanInputError(ValueError):
    """Raised when the provided statistics file is invalid."""


@dataclass(frozen=True)
class PlanResult:
    """Result of generating a scenario specification."""

    spec: Dict[str, Any]
    output_sha256: str


class AnomalyOptionError(ValueError):
    """Raised when anomaly CLI options cannot be parsed."""


_ALLOWED_ANOMALY_TYPES = {"time", "order", "unauth", "token_replay"}
_TIME_MODES = {"propagate", "local"}


def parse_anomaly_options(options: Sequence[str] | None) -> list[Dict[str, Any]]:
    """Parse anomaly CLI options into dictionaries."""

    if not options:
        return []

    return [parse_anomaly_option(option) for option in options]


def parse_anomaly_option(option: str) -> Dict[str, Any]:
    """Parse a single anomaly CLI option string."""

    anomaly_type, params = _extract_option_components(option)
    if anomaly_type not in _ALLOWED_ANOMALY_TYPES:
        raise AnomalyOptionError(f"Unsupported anomaly type: {anomaly_type}")

    if anomaly_type == "time":
        return _build_time_anomaly(params)

    return _build_simple_anomaly(anomaly_type, params)


def _extract_option_components(option: str) -> tuple[str, Dict[str, str]]:
    if not isinstance(option, str) or not option.strip():
        raise AnomalyOptionError("Anomaly option must be a non-empty string")

    text = option.strip()
    if "(" not in text or not text.endswith(")"):
        raise AnomalyOptionError(
            "Anomaly option must follow the pattern type(key=value,...)"
        )

    type_part, params_part = text[:-1].split("(", 1)
    anomaly_type = type_part.strip()
    if not anomaly_type:
        raise AnomalyOptionError("Anomaly option must include an anomaly type")

    params_text = params_part.strip()
    params: Dict[str, str] = {}
    if params_text:
        for token in params_text.split(","):
            token = token.strip()
            if not token:
                continue
            key, separator, value = token.partition("=")
            if not separator:
                raise AnomalyOptionError(
                    f"Invalid parameter format in anomaly option: {token}"
                )
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise AnomalyOptionError(
                    f"Invalid parameter assignment in anomaly option: {token}"
                )
            if key in params:
                raise AnomalyOptionError(
                    f"Duplicate parameter '{key}' in anomaly option"
                )
            params[key] = value

    return anomaly_type, params


def _build_simple_anomaly(
    anomaly_type: str, params: Mapping[str, str]
) -> Dict[str, Any]:
    unexpected = set(params.keys()) - {"p"}
    if unexpected:
        unexpected_fields = ", ".join(sorted(unexpected))
        raise AnomalyOptionError(
            f"{anomaly_type} anomaly does not accept parameter(s): {unexpected_fields}"
        )

    probability = _parse_probability(
        params.get("p"), f"{anomaly_type} anomaly probability"
    )
    return {"type": anomaly_type, "p": probability}


def _build_time_anomaly(params: Mapping[str, str]) -> Dict[str, Any]:
    unexpected = set(params.keys()) - {"mode", "p", "scale", "delta"}
    if unexpected:
        unexpected_fields = ", ".join(sorted(unexpected))
        raise AnomalyOptionError(
            f"time anomaly does not accept parameter(s): {unexpected_fields}"
        )

    mode = params.get("mode")
    if mode is None:
        raise AnomalyOptionError("time anomaly requires mode parameter")
    if mode not in _TIME_MODES:
        valid_modes = ", ".join(sorted(_TIME_MODES))
        raise AnomalyOptionError(f"time anomaly mode must be one of: {valid_modes}")

    probability = _parse_probability(params.get("p"), "time anomaly probability")
    has_scale = "scale" in params
    has_delta = "delta" in params
    if not has_scale and not has_delta:
        raise AnomalyOptionError("time anomaly requires either scale or delta")

    result: Dict[str, Any] = {"type": "time", "mode": mode, "p": probability}
    if has_scale:
        result["scale"] = _parse_positive_float(
            params.get("scale"), "time anomaly scale"
        )
    if has_delta:
        result["delta"] = _parse_float(params.get("delta"), "time anomaly delta")
    return result


def _parse_probability(value: str | None, context: str) -> float:
    probability = _parse_float(value, context)
    if not 0.0 <= probability <= 1.0:
        raise AnomalyOptionError(f"{context} must be between 0 and 1")
    return probability


def _parse_positive_float(value: str | None, context: str) -> float:
    parsed = _parse_float(value, context)
    if parsed <= 0.0:
        raise AnomalyOptionError(f"{context} must be greater than 0")
    return parsed


def _parse_float(value: str | None, context: str) -> float:
    if value is None:
        raise AnomalyOptionError(f"{context} is required")
    try:
        parsed = float(value)
    except ValueError as exc:  # noqa: BLE001 - provide context
        raise AnomalyOptionError(f"{context} must be a number") from exc
    if not math.isfinite(parsed):
        raise AnomalyOptionError(f"{context} must be finite")
    return float(parsed)


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


def build_normal_spec(
    stats: Mapping[str, Any],
    *,
    seed: int,
    anomalies: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
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

    anomalies_list = []
    if anomalies:
        anomalies_list = [dict(anomaly) for anomaly in anomalies]

    spec: Dict[str, Any] = {
        "length": length_raw,
        "users": users_raw,
        "pi": pi_sorted,
        "A": transitions_sorted,
        "dt": {"lognorm": lognorm_sorted},
        "anoms": anomalies_list,
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
    anomalies: Sequence[Mapping[str, Any]] | None = None,
) -> PlanResult:
    """Generate a normal scenario specification."""

    stats = load_stats(stats_path)
    spec = build_normal_spec(stats, seed=seed, anomalies=anomalies)
    output_sha256 = save_spec(spec, output_path)
    return PlanResult(spec=spec, output_sha256=output_sha256)


__all__ = [
    "AnomalyOptionError",
    "PlanInputError",
    "PlanResult",
    "build_normal_spec",
    "load_stats",
    "parse_anomaly_option",
    "parse_anomaly_options",
    "run_plan",
    "save_spec",
]
