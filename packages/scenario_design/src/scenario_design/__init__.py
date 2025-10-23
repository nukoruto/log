"""Scenario design package."""

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("scenario_design")
except PackageNotFoundError:  # pragma: no cover - fallback when not installed
    __version__ = "0.0.0"
