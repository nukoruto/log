"""Log generator package."""

from importlib import metadata


def get_version() -> str:
    """Return the package version."""
    try:
        return metadata.version("log_generator")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]
