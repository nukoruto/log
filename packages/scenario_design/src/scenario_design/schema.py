"""Utilities for validating ``scenario_spec.json`` files."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking import only
    from jsonschema import (  # type: ignore[import-untyped]
        Draft202012Validator as ImportedDraft202012Validator,
    )
    from jsonschema.exceptions import (  # type: ignore[import-untyped]
        ValidationError as ImportedJSONSchemaValidationError,
    )
else:  # pragma: no cover - runtime dependency resolution

    try:
        from jsonschema import (  # type: ignore[import-untyped]
            Draft202012Validator as ImportedDraft202012Validator,
        )
        from jsonschema.exceptions import (  # type: ignore[import-untyped]
            ValidationError as ImportedJSONSchemaValidationError,
        )
    except ModuleNotFoundError:

        class ImportedJSONSchemaValidationError(ValueError):
            """Raised when JSON Schema validation fails."""

        class ImportedDraft202012Validator:
            """Minimal validator implementing the schema rules used in tests."""

            def __init__(self, schema: Mapping[str, Any]):
                self._schema = schema

            def validate(self, instance: Any) -> None:
                try:
                    validate_scenario_spec(instance)
                except ScenarioSpecValidationError as exc:
                    raise ImportedJSONSchemaValidationError(str(exc)) from exc

                _validate_time_anomaly_constraints(instance, self._schema)


def _validate_time_anomaly_constraints(
    instance: Any, schema: Mapping[str, Any]
) -> None:
    """Validate the schema conditional requirements for time anomalies."""

    if not isinstance(instance, Mapping):
        return

    anoms = instance.get("anoms")
    if not isinstance(anoms, list):
        return

    anoms_schema = schema.get("properties", {}).get("anoms", {}).get("items", {})
    if not isinstance(anoms_schema, Mapping):
        return

    conditions = anoms_schema.get("allOf", [])
    if not isinstance(conditions, list) or not conditions:
        return

    for index, anomaly in enumerate(anoms):
        if not isinstance(anomaly, Mapping):
            continue
        for condition in conditions:
            if not isinstance(condition, Mapping):
                continue
            if_schema = condition.get("if")
            then_schema = condition.get("then")
            if not isinstance(if_schema, Mapping) or not isinstance(
                then_schema, Mapping
            ):
                continue
            if _schema_condition_matches(anomaly, if_schema):
                _enforce_then_schema(anomaly, then_schema, index)


def _schema_condition_matches(
    anomaly: Mapping[str, Any], schema: Mapping[str, Any]
) -> bool:
    required = schema.get("required", [])
    if any(field not in anomaly for field in required):
        return False

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return True

    for field, field_schema in properties.items():
        if not isinstance(field_schema, Mapping):
            continue
        if "const" in field_schema and anomaly.get(field) != field_schema["const"]:
            return False

    return True


def _enforce_then_schema(
    anomaly: Mapping[str, Any], schema: Mapping[str, Any], index: int
) -> None:
    required = schema.get("required", [])
    missing = [field for field in required if field not in anomaly]
    if missing:
        missing_fields = ", ".join(missing)
        raise ImportedJSONSchemaValidationError(
            f"anoms[{index}] missing required field(s): {missing_fields}"
        )

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        for option in any_of:
            if not isinstance(option, Mapping):
                continue
            option_required = option.get("required", [])
            if all(field in anomaly for field in option_required):
                break
        else:
            raise ImportedJSONSchemaValidationError(
                f"anoms[{index}] must include either scale or delta"
            )


Draft202012Validator = ImportedDraft202012Validator
JSONSchemaValidationError = ImportedJSONSchemaValidationError

SCENARIO_SPEC_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Scenario specification",
    "type": "object",
    "required": ["length", "users", "pi", "A", "dt", "anoms", "seed"],
    "properties": {
        "length": {"type": "integer", "minimum": 1},
        "users": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer", "minimum": 0},
        "pi": {
            "type": "object",
            "patternProperties": {
                ".+": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "additionalProperties": False,
            "minProperties": 1,
        },
        "A": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
        },
        "dt": {
            "type": "object",
            "required": ["lognorm"],
            "properties": {
                "lognorm": {
                    "type": "object",
                    "required": ["mu", "sigma"],
                    "properties": {
                        "mu": {
                            "type": "object",
                            "minProperties": 1,
                            "additionalProperties": {"type": "number"},
                        },
                        "sigma": {
                            "type": "object",
                            "minProperties": 1,
                            "additionalProperties": {
                                "type": "number",
                                "exclusiveMinimum": 0.0,
                            },
                        },
                    },
                }
            },
        },
        "anoms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "p"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["time", "order", "unauth", "token_replay"],
                    },
                    "mode": {"type": "string", "enum": ["propagate", "local"]},
                    "p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "scale": {"type": "number", "exclusiveMinimum": 0.0},
                    "delta": {"type": "number"},
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"type": {"const": "time"}},
                            "required": ["type"],
                        },
                        "then": {
                            "required": ["mode"],
                            "anyOf": [
                                {"required": ["scale"]},
                                {"required": ["delta"]},
                            ],
                        },
                    }
                ],
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": False,
}


class ScenarioSpecValidationError(ValueError):
    """Raised when a scenario specification violates the schema."""


def validate_scenario_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a scenario specification against :data:`SCENARIO_SPEC_SCHEMA`.

    Args:
        spec: Parsed JSON object representing ``scenario_spec.json``.

    Returns:
        The validated specification for convenience.

    Raises:
        ScenarioSpecValidationError: If the specification violates the schema.
    """

    if not isinstance(spec, Mapping):
        raise ScenarioSpecValidationError("scenario_spec must be a JSON object")

    allowed_fields = set(SCENARIO_SPEC_SCHEMA.get("properties", {}).keys())

    required_fields = ["length", "users", "pi", "A", "dt", "anoms", "seed"]
    for field in required_fields:
        if field not in spec:
            raise ScenarioSpecValidationError(f"Missing required field: {field}")

    unexpected_fields = set(spec.keys()) - allowed_fields
    if unexpected_fields:
        unexpected = ", ".join(sorted(unexpected_fields))
        raise ScenarioSpecValidationError(
            f"Unexpected field(s) in scenario_spec: {unexpected}"
        )

    _validate_positive_integer(spec["length"], "length")
    _validate_positive_integer(spec["users"], "users")
    _validate_non_negative_integer(spec["seed"], "seed")

    pi = spec["pi"]
    categories = _validate_pi(pi)
    transitions = spec["A"]
    _validate_transition_matrix(transitions, categories)

    dt = spec["dt"]
    _validate_timing(dt, categories)

    anoms = spec["anoms"]
    _validate_anomalies(anoms)

    return dict(spec)


def _validate_positive_integer(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScenarioSpecValidationError(f"{field_name} must be a positive integer")


def _validate_non_negative_integer(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScenarioSpecValidationError(
            f"{field_name} must be a non-negative integer"
        )


def _validate_pi(value: Any) -> set[str]:
    if not isinstance(value, Mapping) or not value:
        raise ScenarioSpecValidationError(
            "pi must be a non-empty object of probabilities"
        )

    categories: set[str] = set()
    total = 0.0
    for key, probability in value.items():
        if not isinstance(key, str) or not key:
            raise ScenarioSpecValidationError("pi keys must be non-empty strings")
        if not _is_probability(probability):
            raise ScenarioSpecValidationError(
                f"pi[{key}] must be a probability between 0 and 1"
            )
        categories.add(key)
        total += float(probability)

    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ScenarioSpecValidationError("pi probabilities must sum to 1")

    return categories


def _validate_transition_matrix(value: Any, categories: set[str]) -> None:
    if not isinstance(value, Mapping):
        raise ScenarioSpecValidationError("A must be an object of transition rows")
    if set(value.keys()) != categories:
        raise ScenarioSpecValidationError("A must contain the same states as pi")

    for source, row in value.items():
        if not isinstance(row, Mapping):
            raise ScenarioSpecValidationError(
                f"A[{source}] must be an object of transition probabilities"
            )
        if set(row.keys()) != categories:
            raise ScenarioSpecValidationError(
                f"A[{source}] must contain probabilities for every state"
            )
        row_total = 0.0
        for target, probability in row.items():
            if not _is_probability(probability):
                raise ScenarioSpecValidationError(
                    f"A[{source}][{target}] must be a probability between 0 and 1"
                )
            row_total += float(probability)
        if not math.isclose(row_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ScenarioSpecValidationError(
                f"A[{source}] probabilities must sum to 1"
            )


def _validate_timing(value: Any, categories: set[str]) -> None:
    if not isinstance(value, Mapping) or "lognorm" not in value:
        raise ScenarioSpecValidationError("dt must contain a lognorm specification")
    lognorm = value["lognorm"]
    if not isinstance(lognorm, Mapping):
        raise ScenarioSpecValidationError("dt.lognorm must be an object")

    for key in ["mu", "sigma"]:
        if key not in lognorm:
            raise ScenarioSpecValidationError(f"dt.lognorm must include {key}")

    mu = lognorm["mu"]
    sigma = lognorm["sigma"]
    if not isinstance(mu, Mapping) or set(mu.keys()) != categories:
        raise ScenarioSpecValidationError(
            "dt.lognorm.mu must provide a value for every category"
        )
    if not isinstance(sigma, Mapping) or set(sigma.keys()) != categories:
        raise ScenarioSpecValidationError(
            "dt.lognorm.sigma must provide a value for every category"
        )

    for category in categories:
        mu_value = mu[category]
        sigma_value = sigma[category]
        if not _is_real_number(mu_value):
            raise ScenarioSpecValidationError(
                f"dt.lognorm.mu[{category}] must be a finite number"
            )
        if not _is_positive_real_number(sigma_value):
            raise ScenarioSpecValidationError(
                f"dt.lognorm.sigma[{category}] must be a positive finite number"
            )


def _validate_anomalies(value: Any) -> None:
    if not isinstance(value, list):
        raise ScenarioSpecValidationError("anoms must be an array")

    allowed_types = {"time", "order", "unauth", "token_replay"}
    for index, anomaly in enumerate(value):
        if not isinstance(anomaly, Mapping):
            raise ScenarioSpecValidationError(f"anoms[{index}] must be an object")
        if "type" not in anomaly:
            raise ScenarioSpecValidationError(
                f"anoms[{index}] must include a type field"
            )
        anomaly_type = anomaly["type"]
        if anomaly_type not in allowed_types:
            raise ScenarioSpecValidationError(f"Unknown anomaly type: {anomaly_type}")
        if "p" not in anomaly or not _is_probability(anomaly["p"]):
            raise ScenarioSpecValidationError(
                f"anoms[{index}].p must be a probability between 0 and 1"
            )

        if anomaly_type == "time":
            _validate_time_anomaly(anomaly, index)
        else:
            # For non-time anomalies, ensure no mandatory fields beyond probability
            continue


def _validate_time_anomaly(anomaly: Mapping[str, Any], index: int) -> None:
    mode = anomaly.get("mode")
    if mode not in {"propagate", "local"}:
        raise ScenarioSpecValidationError(
            f"anoms[{index}].mode must be 'propagate' or 'local'"
        )

    scale = anomaly.get("scale")
    delta = anomaly.get("delta")
    if scale is None and delta is None:
        raise ScenarioSpecValidationError(
            f"anoms[{index}] must provide either scale or delta"
        )
    if scale is not None and not _is_positive_real_number(scale):
        raise ScenarioSpecValidationError(
            f"anoms[{index}].scale must be a positive finite number"
        )
    if delta is not None and not _is_real_number(delta):
        raise ScenarioSpecValidationError(
            f"anoms[{index}].delta must be a finite number"
        )


def _is_probability(value: Any) -> bool:
    return _is_real_number(value) and 0.0 <= float(value) <= 1.0


def _is_real_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _is_positive_real_number(value: Any) -> bool:
    return _is_real_number(value) and float(value) > 0.0


__all__ = [
    "SCENARIO_SPEC_SCHEMA",
    "ScenarioSpecValidationError",
    "validate_scenario_spec",
    "Draft202012Validator",
    "JSONSchemaValidationError",
]
