"""Tests for per-package pyproject metadata."""

from __future__ import annotations

from pathlib import Path
import tomllib
import pytest


PACKAGE_CONFIG = {
    "ds_contract": {
        "project_name": "ds-contract",
        "script": {"ds-contract": "ds_contract.cli:main"},
    },
    "scenario_design": {
        "project_name": "scenario-design",
        "script": {"scenario-design": "scenario_design.cli:main"},
    },
    "log_generator": {
        "project_name": "log-generator",
        "script": {"log-generator": "log_generator.cli:cli"},
    },
    "models_lstm": {
        "project_name": "models-lstm",
        "script": {"models-lstm": "models_lstm.cli:main"},
    },
    "matlab_bridge": {
        "project_name": "matlab-bridge",
        "script": {"matlab-bridge": "matlab_bridge.cli:main"},
    },
}


@pytest.mark.parametrize("package", PACKAGE_CONFIG)
def test_pyproject_contains_expected_metadata(package: str) -> None:
    package_dir = Path("packages", package)
    pyproject_path = package_dir / "pyproject.toml"
    assert pyproject_path.exists(), f"missing pyproject.toml for {package}"

    content = tomllib.loads(pyproject_path.read_text())

    build_system = content.get("build-system")
    assert build_system is not None, "build-system section is required"
    assert build_system.get("build-backend") == "setuptools.build_meta"
    requires = build_system.get("requires", [])
    assert any(req.startswith("setuptools") for req in requires)

    project = content.get("project")
    assert project is not None, "project section is required"
    config = PACKAGE_CONFIG[package]
    assert project.get("name") == config["project_name"]
    assert project.get("version") == "0.1.0"
    assert project.get("requires-python") == ">=3.11"

    scripts = project.get("scripts")
    assert scripts is not None, "project.scripts section is required"
    assert scripts == config["script"]

    setuptools_config = content.get("tool", {}).get("setuptools", {})
    find_config = setuptools_config.get("packages", {}).get("find")
    assert find_config is not None, "[tool.setuptools.packages.find] section is required"
    assert find_config.get("where") == ["src"], "packages must be discovered from src"
