"""Command line interface for the scenario-design package."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fit import ALGO_VERSION, estimate_statistics


def _emit_log(event: str, payload: dict[str, Any]) -> None:
    message = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _handle_fit(args: argparse.Namespace) -> None:
    input_path = Path(args.deltified)
    output_path = Path(args.out)

    start = time.time()
    _emit_log(
        "start",
        {
            "command": "fit",
            "input": str(input_path),
            "output": str(output_path),
            "algo_version": ALGO_VERSION,
        },
    )

    try:
        stats = estimate_statistics(input_path)
    except Exception as exc:  # pragma: no cover - defensive branch
        _emit_log("error", {"command": "fit", "message": str(exc)})
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(stats.to_dict(), handle)

    duration = time.time() - start
    _emit_log(
        "complete",
        {
            "command": "fit",
            "input": str(input_path),
            "output": str(output_path),
            "input_sha256": stats.input_sha256,
            "algo_version": stats.algo_version,
            "duration_seconds": duration,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenario-design")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit", help="Estimate scenario statistics from deltified.csv"
    )
    fit_parser.add_argument(
        "deltified", help="Path to deltified.csv produced by ds-contract"
    )
    fit_parser.add_argument(
        "--out", required=True, help="Destination path for stats.pkl"
    )
    fit_parser.set_defaults(func=_handle_fit)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        handler = args.func
    except AttributeError:  # pragma: no cover - defensive branch
        parser.print_help()
        raise SystemExit(1)

    handler(args)


__all__ = ["build_parser", "main"]
