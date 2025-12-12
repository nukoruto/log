"""Command-line interface for the matlab-bridge package."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .export import ExportError, export_to_mat


class ParserError(Exception):
    """Raised when CLI argument parsing fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _StructuredArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises an exception instead of exiting on errors."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise ParserError(message)


ALGO_VERSION = "matlab-bridge/1.0.0"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``matlab-bridge`` CLI."""

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except ParserError as exc:
        return _handle_parser_error(argv or [], exc)

    if args.command == "export":
        return _run_export(
            Path(args.input),
            Path(args.output),
            Path(args.meta) if args.meta is not None else None,
            args.seed,
        )

    parser.error("No command specified")
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = _StructuredArgumentParser(
        prog="matlab-bridge",
        description="Export scored.csv files to MATLAB MAT format.",
    )
    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser(
        "export",
        help="Convert scored.csv into ref.mat",
    )
    export_parser.add_argument(
        "--in", dest="input", required=True, help="Path to scored.csv"
    )
    export_parser.add_argument(
        "--out", dest="output", required=True, help="Destination MAT file"
    )
    export_parser.add_argument(
        "--meta",
        dest="meta",
        help="Optional path to write the export metadata JSON",
    )
    export_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        required=True,
        help="Deterministic seed for the export pipeline",
    )

    return parser


def _handle_parser_error(argv: list[str], error: ParserError) -> int:
    payload: dict[str, Any] = {
        "error": "argument_error",
        "message": error.message,
        "algo_version": ALGO_VERSION,
    }

    _log_event("export.error", payload)
    return 1


def _run_export(
    input_path: Path, output_path: Path, meta_path: Path | None, seed: int
) -> int:
    _set_deterministic_seed(seed)
    base_context: dict[str, Any] = {
        "seed": seed,
        "algo_version": ALGO_VERSION,
    }
    if meta_path is not None:
        base_context["meta"] = str(meta_path)

    input_sha256: str | None
    hash_error: OSError | None = None
    error_code: str | None = None
    try:
        input_sha256 = _compute_file_sha256(input_path)
    except FileNotFoundError as exc:
        input_sha256 = None
        hash_error = exc
        error_code = "file_not_found"
    except OSError as exc:  # pragma: no cover - rare permissions issues
        input_sha256 = None
        hash_error = exc
        error_code = "io_error"

    _log_event(
        "export.start",
        {
            "input": str(input_path),
            "output": str(output_path),
            "input_sha256": input_sha256,
            **base_context,
        },
    )

    if hash_error is not None:
        _log_event(
            "export.error",
            {
                "error": error_code,
                "message": str(hash_error),
                "input_sha256": input_sha256,
                **base_context,
            },
        )
        return 1

    try:
        result = export_to_mat(
            input_path,
            output_path,
            meta_path=meta_path,
            meta_context=(
                {
                    "seed": seed,
                    "algo_version": ALGO_VERSION,
                    "output": str(output_path),
                }
                if meta_path is not None
                else None
            ),
        )
    except FileNotFoundError as exc:
        _log_event(
            "export.error",
            {
                "error": "file_not_found",
                "message": str(exc),
                "input_sha256": input_sha256,
                **base_context,
            },
        )
        return 1
    except ExportError as exc:
        _log_event(
            "export.error",
            {
                "error": "validation_error",
                "message": str(exc),
                "input_sha256": input_sha256,
                **base_context,
            },
        )
        return 1

    _log_event(
        "export.complete",
        {
            "rows": result.row_count,
            "input_sha256": result.input_sha256,
            "output": str(output_path),
            "generated_at": result.generated_at.isoformat().replace("+00:00", "Z"),
            **base_context,
        },
    )
    return 0


def _log_event(event: str, payload: dict[str, Any]) -> None:
    entry = {
        "event": event,
        "timestamp": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
    }
    entry.update(payload)
    json.dump(entry, sys.stdout)
    sys.stdout.write("\n")


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - numpy optional
        pass
    else:
        np.random.seed(seed)

    try:  # pragma: no cover - torch optional
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except AttributeError:
        pass


def _compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
