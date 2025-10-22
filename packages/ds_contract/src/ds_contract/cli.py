"""Command-line interface for the ds_contract package."""

from __future__ import annotations

import argparse
from typing import Sequence

from ds_contract import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="ds-contract",
        description="Tools for validating and transforming contract CSV datasets.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ds-contract {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ds-contract console script."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
