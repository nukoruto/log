"""Tests for scenario specification schema validation."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

package_root = Path(__file__).resolve().parents[1] / "src"
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

try:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    from jsonschema.exceptions import (  # type: ignore[import-untyped]
        ValidationError as JSONSchemaValidationError,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in CI without jsonschema
    from scenario_design.schema import (  # type: ignore[attr-defined]
        Draft202012Validator,
        JSONSchemaValidationError,
    )

from scenario_design.schema import (  # noqa: E402
    SCENARIO_SPEC_SCHEMA,
    ScenarioSpecValidationError,
    validate_scenario_spec,
)


@pytest.fixture()
def minimal_spec() -> dict[str, Any]:
    """Return a minimal valid scenario specification."""

    return {
        "length": 512,
        "users": 10,
        "pi": {"AUTH": 0.3, "READ": 0.6, "UPDATE": 0.1},
        "A": {
            "AUTH": {"AUTH": 0.1, "READ": 0.8, "UPDATE": 0.1},
            "READ": {"AUTH": 0.2, "READ": 0.6, "UPDATE": 0.2},
            "UPDATE": {"AUTH": 0.3, "READ": 0.2, "UPDATE": 0.5},
        },
        "dt": {
            "lognorm": {
                "mu": {"AUTH": 0.1, "READ": 0.5, "UPDATE": 0.7},
                "sigma": {"AUTH": 0.3, "READ": 0.4, "UPDATE": 0.5},
            }
        },
        "anoms": [
            {"type": "time", "mode": "propagate", "p": 0.02, "scale": 3.0},
            {"type": "order", "p": 0.01},
        ],
        "seed": 42,
    }


def test_minimal_spec_is_valid(minimal_spec: dict[str, Any]) -> None:
    """A minimal specification must validate without raising."""

    validate_scenario_spec(minimal_spec)


def test_missing_required_field(minimal_spec: dict[str, Any]) -> None:
    """Omitting a required field should raise a validation error."""

    minimal_spec.pop("A")
    with pytest.raises(ScenarioSpecValidationError, match="Missing required field: A"):
        validate_scenario_spec(minimal_spec)


def test_invalid_type_rejected(minimal_spec: dict[str, Any]) -> None:
    """Invalid field types should trigger a validation error."""

    minimal_spec["length"] = "512"  # type: ignore[assignment]
    with pytest.raises(
        ScenarioSpecValidationError, match="length must be a positive integer"
    ):
        validate_scenario_spec(minimal_spec)


def test_unexpected_top_level_field_rejected(minimal_spec: dict[str, Any]) -> None:
    """Additional top-level fields must not be accepted."""

    minimal_spec["unexpected"] = "value"
    with pytest.raises(
        ScenarioSpecValidationError,
        match=r"Unexpected field\(s\) in scenario_spec",
    ):
        validate_scenario_spec(minimal_spec)


@pytest.fixture()
def schema_validator() -> Draft202012Validator:
    """Return a Draft 2020-12 validator for the scenario schema."""

    return Draft202012Validator(SCENARIO_SPEC_SCHEMA)


def test_time_anomaly_requires_mode(
    minimal_spec: dict[str, Any], schema_validator: Draft202012Validator
) -> None:
    """Time anomalies without a mode should be rejected by all validators."""

    minimal_spec["anoms"][0].pop("mode")

    with pytest.raises(
        ScenarioSpecValidationError,
        match="anoms\\[0\\].mode must be 'propagate' or 'local'",
    ):
        validate_scenario_spec(minimal_spec)

    with pytest.raises(JSONSchemaValidationError, match="mode"):
        schema_validator.validate(minimal_spec)


def test_time_anomaly_requires_scale_or_delta(
    minimal_spec: dict[str, Any], schema_validator: Draft202012Validator
) -> None:
    """Time anomalies must provide either scale or delta."""

    anomaly = minimal_spec["anoms"][0]
    anomaly.pop("scale", None)
    anomaly.pop("delta", None)

    with pytest.raises(
        ScenarioSpecValidationError,
        match="anoms\\[0\\] must provide either scale or delta",
    ):
        validate_scenario_spec(minimal_spec)

    with pytest.raises(JSONSchemaValidationError, match="(scale|delta)"):
        schema_validator.validate(minimal_spec)


def test_fallback_validator_enforces_required_fields(
    minimal_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """フォールバックバリデータでも必須項目を検証する。"""

    import builtins

    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        if name.startswith("jsonschema"):
            raise ModuleNotFoundError
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "scenario_design_schema_fallback"
    schema_path = package_root / "scenario_design" / "schema.py"
    spec = importlib.util.spec_from_file_location(module_name, schema_path)
    assert spec and spec.loader is not None
    fallback_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback_module)

    fallback_validator = fallback_module.Draft202012Validator(
        fallback_module.SCENARIO_SPEC_SCHEMA
    )

    missing_type = copy.deepcopy(minimal_spec)
    missing_type["anoms"][0].pop("type")
    with pytest.raises(fallback_module.JSONSchemaValidationError, match="type"):
        fallback_validator.validate(missing_type)

    missing_probability = copy.deepcopy(minimal_spec)
    missing_probability["anoms"][0].pop("p")
    with pytest.raises(fallback_module.JSONSchemaValidationError, match="p"):
        fallback_validator.validate(missing_probability)

    extra_field = copy.deepcopy(minimal_spec)
    extra_field["unexpected"] = True
    with pytest.raises(
        fallback_module.JSONSchemaValidationError,
        match=r"Unexpected field\(s\) in scenario_spec",
    ):
        fallback_validator.validate(extra_field)

    monkeypatch.setattr(builtins, "__import__", original_import)
