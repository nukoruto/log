"""Command line interface for the log generator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .generate import (
    compute_spec_sha256,
    format_utc,
    generate_normal_records,
    load_spec,
    write_contract_csv,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""

    parser = argparse.ArgumentParser(prog="log-generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate contract-compliant logs")
    run_parser.add_argument(
        "--spec", type=Path, required=True, help="Path to scenario_spec.json"
    )
    run_parser.add_argument(
        "--seed", type=int, required=True, help="Deterministic seed"
    )
    run_parser.add_argument(
        "--normal", type=Path, required=True, help="Output path for normal.csv"
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _command_run(args)
    parser.error("unknown command")
    return 2


def _command_run(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    _emit_log(
        {
            "event": "log_generator.run.start",
            "seed": args.seed,
            "spec": str(args.spec),
            "normal": str(args.normal),
            "timestamp": _now_utc(),
        }
    )

    spec = load_spec(args.spec)
    spec_sha = compute_spec_sha256(args.spec)
    records = generate_normal_records(spec, args.seed)
    write_contract_csv(records, args.normal)

    duration = time.perf_counter() - start
    _emit_log(
        {
            "event": "log_generator.run.complete",
            "seed": args.seed,
            "spec": str(args.spec),
            "spec_sha256": spec_sha,
            "records": len(records),
            "algo_version": spec.algo_version,
            "normal": str(args.normal),
            "timestamp": _now_utc(),
            "duration_seconds": round(duration, 6),
        }
    )
    return 0


def _emit_log(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _now_utc() -> str:
    return format_utc(datetime.now(timezone.utc))


if __name__ == "__main__":
    raise SystemExit(main())
