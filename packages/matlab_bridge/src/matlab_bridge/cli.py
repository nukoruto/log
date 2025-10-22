"""Command line interface for MATLAB bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as _np  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _np = None

from .export import ExportError, ExportSummary, export_to_mat


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="matlab-bridge",
        description="Export scored CSV files to MATLAB-compatible MAT files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Convert a scored CSV file into a MATLAB MAT file.",
    )
    export_parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Path to the input scored CSV file.",
    )
    export_parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Destination path for the MATLAB MAT file.",
    )
    export_parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for deterministic operations.",
    )
    return parser


def _compute_sha256(path: Path) -> str:
    """Compute the SHA256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(payload: dict[str, Any]) -> None:
    """Emit a JSON log payload to stdout."""
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _seed_everything(seed: int) -> None:
    """Seed all supported random number generators."""
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed)


def _run_export(args: argparse.Namespace) -> int:
    """Execute the export subcommand."""
    seed = args.seed
    _seed_everything(seed)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    input_sha = None
    if input_path.exists():
        input_sha = _compute_sha256(input_path)

    start_payload: dict[str, Any] = {
        "event": "start",
        "command": "export",
        "seed": seed,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_sha256": input_sha,
    }
    _log(start_payload)

    try:
        summary: ExportSummary = export_to_mat(input_path, output_path)
    except (FileNotFoundError, ExportError) as exc:
        error_payload: dict[str, Any] = {
            "event": "error",
            "command": "export",
            "seed": seed,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_sha256": input_sha,
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }
        _log(error_payload)
        return 1

    complete_payload: dict[str, Any] = {
        "event": "complete",
        "command": "export",
        "seed": seed,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_sha256": input_sha,
        "rows": summary.row_count,
        "t_start": summary.t_start,
        "t_end": summary.t_end,
    }
    _log(complete_payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the matlab-bridge CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "export":
        return _run_export(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2  # pragma: no cover - argparse raises before this point


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
