"""Command line interface for scenario design."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import __version__
from .fit import compute_file_sha256, run_fit, seed_everything
from .plan import (
    AnomalyOptionError,
    PlanInputError,
    PlanResult,
    parse_anomaly_options,
    run_plan,
)
from .schema import ScenarioSpecValidationError


class ArgumentParserError(Exception):
    """Raised when CLI argument parsing fails."""


class ScenarioDesignArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of exiting on errors."""

    def error(self, message: str) -> None:  # type: ignore[override]
        # pragma: no cover - exercised via tests
        raise ArgumentParserError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = ScenarioDesignArgumentParser(
        prog="scenario-design", description="Scenario design utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit", help="Estimate Markov and timing statistics"
    )
    fit_parser.add_argument("deltified", type=Path, help="Path to deltified.csv")
    fit_parser.add_argument(
        "--out", type=Path, required=True, help="Where to store stats.pkl"
    )
    fit_parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used for deterministic estimation",
    )

    plan_parser = subparsers.add_parser(
        "plan", help="Generate a normal scenario specification"
    )
    plan_parser.add_argument(
        "--stats",
        type=Path,
        required=True,
        help="Path to stats.pkl produced by scenario-design fit",
    )
    plan_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to store scenario_spec.json",
    )
    plan_parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for deterministic scenario generation",
    )
    plan_parser.add_argument(
        "--anom",
        action="append",
        default=[],
        metavar="TYPE(args)",
        help=(
            "Anomaly specification, e.g. time(mode=propagate,p=0.02,scale=3.0). "
            "May be provided multiple times."
        ),
    )

    return parser


def log_event(event: str, **payload: Any) -> None:
    record: Dict[str, Any] = {"event": event, "version": __version__}
    record.update(payload)
    print(json.dumps(record, ensure_ascii=False))


def _normalize_argv(argv: Iterable[str] | None) -> List[str]:
    if argv is None:
        return list(sys.argv[1:])
    return list(argv)


def _determine_fit_error_code(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return "SCENARIO_DESIGN_CONTRACT_ERROR"
    return "SCENARIO_DESIGN_FIT_ERROR"


def _determine_plan_error_code(exc: Exception) -> str:
    if isinstance(exc, (PlanInputError, ScenarioSpecValidationError, ValueError)):
        return "SCENARIO_DESIGN_CONTRACT_ERROR"
    return "SCENARIO_DESIGN_PLAN_ERROR"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = _normalize_argv(argv)
    try:
        args = parser.parse_args(argv_list)
    except ArgumentParserError as exc:
        command = argv_list[0] if argv_list else None
        log_event(
            "error",
            command=command,
            message=str(exc),
            error_code="SCENARIO_DESIGN_ARGUMENT_ERROR",
            seed=None,
            input=None,
            output=None,
            input_sha256=None,
        )
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "fit":
        input_path: Path = args.deltified
        output_path: Path = args.out
        seed: int = args.seed
        seed_everything(seed)
        input_sha256: str | None = None
        if input_path.exists():
            try:
                input_sha256 = compute_file_sha256(input_path)
            except OSError:
                input_sha256 = None
        log_event(
            "start",
            command="fit",
            input=str(input_path),
            output=str(output_path),
            seed=seed,
            input_sha256=input_sha256,
        )
        try:
            stats = run_fit(
                input_path,
                output_path,
                seed=seed,
                input_sha256=input_sha256,
            )
        except Exception as exc:  # noqa: BLE001 - propagate as CLI failure
            error_code = _determine_fit_error_code(exc)
            log_event(
                "error",
                command="fit",
                input=str(input_path),
                output=str(output_path),
                message=str(exc),
                error_code=error_code,
                seed=seed,
                input_sha256=input_sha256,
            )
            print(str(exc), file=sys.stderr)
            return 1
        log_event(
            "complete",
            command="fit",
            input=str(input_path),
            output=str(output_path),
            n_events=stats["n_events"],
            n_sessions=stats["n_sessions"],
            input_sha256=stats["input_sha256"],
            seed=seed,
        )
        return 0

    if args.command == "plan":
        stats_path: Path = args.stats
        output_path = args.out
        seed = args.seed
        try:
            anomalies = parse_anomaly_options(args.anom)
        except AnomalyOptionError as exc:
            log_event(
                "error",
                command="plan",
                input=str(stats_path),
                output=str(output_path),
                message=str(exc),
                error_code="SCENARIO_DESIGN_ARGUMENT_ERROR",
                seed=seed,
                input_sha256=None,
            )
            print(str(exc), file=sys.stderr)
            return 2
        seed_everything(seed)
        stats_sha256: str | None = None
        if stats_path.exists():
            try:
                stats_sha256 = compute_file_sha256(stats_path)
            except OSError:
                stats_sha256 = None
        log_event(
            "start",
            command="plan",
            input=str(stats_path),
            output=str(output_path),
            seed=seed,
            input_sha256=stats_sha256,
        )
        try:
            result: PlanResult = run_plan(
                stats_path,
                output_path,
                seed=seed,
                anomalies=anomalies,
            )
        except Exception as exc:  # noqa: BLE001 - propagate as CLI failure
            error_code = _determine_plan_error_code(exc)
            log_event(
                "error",
                command="plan",
                input=str(stats_path),
                output=str(output_path),
                message=str(exc),
                error_code=error_code,
                seed=seed,
                input_sha256=stats_sha256,
            )
            print(str(exc), file=sys.stderr)
            return 1
        log_event(
            "complete",
            command="plan",
            input=str(stats_path),
            output=str(output_path),
            length=result.spec["length"],
            users=result.spec["users"],
            input_sha256=stats_sha256,
            output_sha256=result.output_sha256,
            seed=seed,
        )
        return 0

    log_event("error", message="Unknown command")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
