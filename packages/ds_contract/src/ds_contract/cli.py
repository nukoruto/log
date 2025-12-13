"""Data contract CLI implementation for validate/sessionize/deltify."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import hmac
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .contract import CONTRACT_COLUMNS, REQUIRED_CONTRACT_FIELDS, ensure_required_fields
from .dt import DELTIFIED_COLUMNS, deltify_session_rows
from .sessionize import (
    EPSILON,
    SESSION_COLUMNS,
    sessionize_contract,
)
from .ait import AitLogProcessor


class CommandError(RuntimeError):
    """Raised when CLI input validation fails."""

    def __init__(
        self, code: str, message: str, *, hint: str | None = None, exit_code: int = 1
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.exit_code = exit_code


@dataclass(slots=True)
class JsonLogger:
    """Utility emitting JSON structured logs to stdout."""

    command: str
    seed: int | None

    def emit(self, *, event: str, **payload: object) -> None:
        record: dict[str, object] = {
            "event": event,
            "command": self.command,
            "seed": self.seed,
        }
        for key, value in payload.items():
            record[key] = value
        print(json.dumps(record, ensure_ascii=False))

    def log_start(self, details: dict[str, object]) -> None:
        self.emit(event="start", details=details)

    def log_complete(self, details: dict[str, object]) -> None:
        self.emit(event="complete", details=details)

    def log_error(self, *, code: str, message: str, hint: str | None = None) -> None:
        payload: dict[str, object] = {"code": code, "message": message}
        if hint:
            payload["hint"] = hint
        self.emit(event="error", **payload)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        prog="ds-contract", description="Data contract processing CLI"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Deterministic seed applied across all subcommands",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    _register_validate(subparsers)
    _register_sessionize(subparsers)
    _register_deltify(subparsers)
    _register_process_ait(subparsers)

    return parser


def _register_validate(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "validate", help="Normalize raw CSV into contract.csv"
    )
    parser.add_argument("input_csv", help="Raw CSV to validate")
    parser.add_argument(
        "--map", dest="mapping", required=True, help="YAML column mapping file"
    )
    parser.add_argument("--out", required=True, help="Contract CSV output path")
    parser.add_argument(
        "--meta",
        help="Metadata JSON output path (default: <out>.meta.json)",
    )
    parser.set_defaults(handler=_handle_validate)


def _register_sessionize(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("sessionize", help="Derive sessions and Δt gaps")
    parser.add_argument("contract_csv", help="Validated contract CSV input")
    parser.add_argument("--out", required=True, help="Sessionized CSV output path")
    parser.add_argument(
        "--meta", required=True, help="Session metadata JSON output path"
    )
    parser.set_defaults(handler=_handle_sessionize)


def _register_deltify(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("deltify", help="Compute robust Δt features")
    parser.add_argument("sessioned_csv", help="Sessionized CSV input")
    parser.add_argument("--out", required=True, help="Deltified CSV output path")
    parser.add_argument("--meta", required=True, help="Δt metadata JSON output path")
    parser.set_defaults(handler=_handle_deltify)


def _register_process_ait(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("process-ait", help="Process AIT logs and labels")
    parser.add_argument("log_file", help="Input Apache Combined Log file")
    parser.add_argument("label_file", help="Input Label CSV file")
    parser.add_argument("--out", required=True, help="Output prefix for CSV files")
    parser.set_defaults(handler=_handle_process_ait)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0 and argv and any(opt in argv for opt in {"--help", "-h"}):
            return 0
        raise

    if getattr(args, "handler", None) is None:
        parser.print_help()
        return 0

    logger = JsonLogger(command=args.command, seed=getattr(args, "seed", None))

    if getattr(args, "seed", None) is None:
        logger.log_error(
            code="MISSING_SEED",
            message="--seed is required for deterministic processing.",
            hint="Invoke the command with an explicit --seed integer value.",
        )
        return 1

    _set_global_seed(args.seed)

    try:
        result = args.handler(args, logger)
    except CommandError as exc:
        logger.log_error(code=exc.code, message=exc.message, hint=exc.hint)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.log_error(code="UNEXPECTED", message=str(exc))
        return 1
    else:
        logger.log_complete(result)
        return 0


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        return


def _default_meta_path(output_path: Path) -> Path:
    stem = output_path.stem or output_path.name or "contract"
    return output_path.with_name(f"{stem}.meta.json")


def _handle_validate(args: argparse.Namespace, logger: JsonLogger) -> dict[str, object]:
    input_path = Path(args.input_csv)
    mapping_path = Path(args.mapping)
    output_path = Path(args.out)
    meta_arg = getattr(args, "meta", None)
    if meta_arg:
        meta_path = Path(meta_arg)
    else:
        meta_path = _default_meta_path(output_path)

    logger.log_start(
        {
            "input": str(input_path),
            "mapping": str(mapping_path),
            "output": str(output_path),
            "meta": str(meta_path),
        }
    )

    if not input_path.exists():
        raise CommandError(
            "INPUT_NOT_FOUND",
            f"Input CSV '{input_path}' does not exist.",
            hint="Provide a valid raw CSV file for validation.",
        )

    if not mapping_path.exists():
        raise CommandError(
            "MAPPING_NOT_FOUND",
            f"Mapping YAML '{mapping_path}' does not exist.",
            hint="Pass --map with a column mapping YAML file.",
        )

    mapping = _load_mapping(mapping_path)
    uid_key = str(args.seed).encode("utf-8")
    rows = _read_and_normalize_contract(input_path, mapping, uid_key)
    _write_csv(output_path, CONTRACT_COLUMNS, rows)

    meta = {
        "seed": args.seed,
        "input_sha256": _compute_sha256(input_path),
        "row_count": len(rows),
        "mapping": mapping,
        "required_columns": sorted(REQUIRED_CONTRACT_FIELDS),
        "required_columns": sorted(REQUIRED_CONTRACT_FIELDS),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "row_count": len(rows),
        "input_sha256": meta["input_sha256"],
    }


def _handle_sessionize(
    args: argparse.Namespace, logger: JsonLogger
) -> dict[str, object]:
    input_path = Path(args.contract_csv)
    output_path = Path(args.out)
    meta_path = Path(args.meta)

    logger.log_start(
        {"input": str(input_path), "output": str(output_path), "meta": str(meta_path)}
    )

    rows = _load_contract_rows(input_path)
    if not rows:
        raise CommandError(
            "EMPTY_CONTRACT",
            "Contract CSV contains no rows to sessionize.",
            hint="Run validate on a non-empty dataset before sessionize.",
        )

    result = sessionize_contract(rows)
    sessionized = result.rows

    _write_csv(output_path, SESSION_COLUMNS, sessionized)

    meta = {
        "seed": args.seed,
        "input_sha256": _compute_sha256(input_path),
        "method": result.method,
        "threshold_seconds": result.threshold_seconds,
        "log_threshold": result.log_threshold,
        "histogram": result.histogram,
        "bins": result.bins,
        "within_class_variance_ratio": result.within_class_variance_ratio,
        "epsilon": EPSILON,
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "row_count": len(sessionized),
        "threshold_seconds": result.threshold_seconds,
        "method": result.method,
        "input_sha256": meta["input_sha256"],
    }


def _handle_deltify(args: argparse.Namespace, logger: JsonLogger) -> dict[str, object]:
    input_path = Path(args.sessioned_csv)
    output_path = Path(args.out)
    meta_path = Path(args.meta)

    logger.log_start(
        {"input": str(input_path), "output": str(output_path), "meta": str(meta_path)}
    )

    rows = _load_sessioned_rows(input_path)
    if not rows:
        raise CommandError(
            "EMPTY_SESSIONED",
            "Sessioned CSV contains no rows for Δt feature extraction.",
            hint="Run sessionize before deltify on a populated dataset.",
        )

    result = deltify_session_rows(rows)
    meta = dict(result.meta)
    meta.update(
        {
            "seed": args.seed,
            "input_sha256": _compute_sha256(input_path),
        }
    )
    _write_csv(output_path, DELTIFIED_COLUMNS, result.rows)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "row_count": len(result.rows),
        "fallback_users": meta.get("fallback_users", []),
        "input_sha256": meta["input_sha256"],
    }


def _handle_process_ait(
    args: argparse.Namespace, logger: JsonLogger
) -> dict[str, object]:
    log_path = Path(args.log_file)
    label_path = Path(args.label_file)
    output_prefix = args.out

    logger.log_start(
        {
            "log_file": str(log_path),
            "label_file": str(label_path),
            "output_prefix": output_prefix,
        }
    )

    if not log_path.exists():
        raise CommandError("LOG_NOT_FOUND", f"Log file '{log_path}' not found.")
    if not label_path.exists():
        raise CommandError("LABEL_NOT_FOUND", f"Label file '{label_path}' not found.")

    processor = AitLogProcessor(log_path, label_path)
    result = processor.process(output_prefix)

    return result


def _load_mapping(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = _parse_simple_yaml(text)

    mapping = loaded
    if not isinstance(mapping, dict):
        raise CommandError(
            "INVALID_MAPPING",
            "Mapping YAML must define a dictionary from contract columns to input columns.",
            hint="Ensure the YAML has key: value pairs for each contract column.",
        )

    normalized: dict[str, str] = {}
    for key, value in mapping.items():
        if key not in CONTRACT_COLUMNS and key not in (
            "ip",
            "user_agent",
            "sid",
            "uid_raw",
            "sub",
            "dc",
        ):
            raise CommandError(
                "UNKNOWN_MAPPING_KEY",
                f"Unsupported contract column '{key}' in mapping file.",
                hint="Use only columns defined in the contract schema or ID generation keys (sid, uid_raw, sub, dc, ip, user_agent).",
            )
        if not isinstance(value, str):
            raise CommandError(
                "INVALID_MAPPING_VALUE",
                f"Mapping for '{key}' must be a string column name.",
            )
        normalized[key] = value

    missing = set(ensure_required_fields(normalized.keys()))
    # If any ID generation source is mapped, uid is not required in the mapping
    id_sources = {"sid", "uid_raw", "sub", "dc", "ip", "user_agent"}
    # Note: ip/user_agent condition was stricter (both needed), but new logic allows fallback.
    # We will assume if ANY ID source is present, we can TRY to generate.
    # Strictly speaking, ip+ua is the last fallback. 
    # For now, if any of these are present, we waive the explicit 'uid' requirement.
    if (set(normalized.keys()) & id_sources) and "uid" in missing:
        missing.remove("uid")
        
    required_missing = frozenset(missing)
    if required_missing:
        raise CommandError(
            "MISSING_MAPPING",
            f"Mapping file lacks required contract columns: {sorted(required_missing)}.",
            hint="Add source column mappings for all required contract fields.",
        )
    return normalized


def _parse_simple_yaml(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise CommandError(
                "INVALID_MAPPING_LINE",
                f"Unable to parse mapping line '{raw_line}'.",
                hint="Use 'key: value' pairs in the mapping file.",
            )
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        mapping[key] = value
    return mapping


def _read_and_normalize_contract(
    input_path: Path, mapping: dict[str, str], uid_key: bytes
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    previous_timestamp: str | None = None

    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        # Check only for REQUIRED columns in the mapping
        missing_columns = [
            source
            for col, source in mapping.items()
            if col in REQUIRED_CONTRACT_FIELDS and source not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise CommandError(
                "MISSING_INPUT_COLUMNS",
                f"Input CSV is missing mapped columns: {missing_columns}",
                hint="Verify the raw CSV headers and the mapping YAML entries.",
            )

        for index, raw_row in enumerate(reader, start=1):
            contract_row: dict[str, str] = {}
            for column in CONTRACT_COLUMNS:
                source_column = mapping.get(column)
                value = raw_row.get(source_column, "") if source_column else ""
                
                if column in REQUIRED_CONTRACT_FIELDS and not value:
                    # Exception: uid can be missing if we have any source to generate it
                    id_sources = {"sid", "uid_raw", "sub", "dc", "ip", "user_agent"}
                    # Ideally we check if specific meaningful sources are mapped, 
                    # but for now presence of any ID key in mapping is enough to defer check.
                    # Note: ip+ua is a pair, but simplification "set intersection" catches "ip" or "user_agent".
                    if column == "uid" and (set(mapping.keys()) & id_sources):
                        pass
                    else:
                        raise CommandError(
                            "MISSING_REQUIRED_VALUE",
                            f"Row {index} is missing required column '{source_column}'.",
                            hint="Ensure required columns are populated in the raw CSV.",
                        )
                
                if column == "timestamp_utc":
                    contract_row[column] = _normalize_timestamp(value)
                else:
                    contract_row[column] = value

            # Derive op_category if missing
            if not contract_row.get("op_category"):
                contract_row["op_category"] = contract_row.get("method", "UNKNOWN")

            current_timestamp = contract_row["timestamp_utc"]
            if (
                previous_timestamp is not None
                and current_timestamp < previous_timestamp
            ):
                raise CommandError(
                    "REVERSED_ORDER",
                    (
                        "Input timestamps must be non-decreasing; "
                        f"row {index} timestamp {current_timestamp} "
                        f"is earlier than previous {previous_timestamp}."
                    ),
                    hint=(
                        "Sort the raw CSV by timestamp_utc ascending before running validate."
                    ),
                )
            previous_timestamp = current_timestamp

            # UID handling: prioritized base_id generation
            # Priority: sid > uid_raw > sub > dc > (ip+ua)
            
            raw_sid = raw_row.get(mapping.get("sid", ""), "")
            raw_uid_raw = raw_row.get(mapping.get("uid_raw", ""), "")
            raw_sub = raw_row.get(mapping.get("sub", ""), "")
            raw_dc = raw_row.get(mapping.get("dc", ""), "")
            raw_ip = raw_row.get(mapping.get("ip", ""), "")
            raw_ua = raw_row.get(mapping.get("user_agent", ""), "")

            base_id = ""

            if raw_sid:
                base_id = raw_sid
                # If sid is available, we also populate session_id if it wasn't mapped/provided
                if not contract_row.get("session_id"):
                    contract_row["session_id"] = raw_sid
            elif raw_uid_raw:
                base_id = hmac.new(uid_key, raw_uid_raw.encode("utf-8"), hashlib.sha256).hexdigest()
            elif raw_sub:
                base_id = hmac.new(uid_key, raw_sub.encode("utf-8"), hashlib.sha256).hexdigest()
            elif raw_dc:
                base_id = hmac.new(uid_key, raw_dc.encode("utf-8"), hashlib.sha256).hexdigest()
            elif raw_ip and raw_ua:
                uid_message = f"{raw_ip}|{raw_ua}".encode("utf-8")
                base_id = hmac.new(uid_key, uid_message, hashlib.sha256).hexdigest()
            
            # If we calculated a base_id, assign it to uid if not already present
            if base_id and not contract_row.get("uid"):
                contract_row["uid"] = base_id
            
            # Final check (retained from previous logic)
            if not contract_row.get("uid"):
                 raise CommandError(
                    "MISSING_UID_SOURCE",
                    f"Row {index} has no 'uid' and no suitable source (sid, uid_raw, sub, dc, ip+ua).",
                    hint="Provide at least one ID source in the mapping.",
                )
            
            rows.append(contract_row)
    return rows


def _normalize_timestamp(value: str) -> str:
    try:
        if value.isdigit():
            timestamp_ms = int(value)
            if len(value) > 10:
                seconds = timestamp_ms / 1000
            else:
                seconds = float(timestamp_ms)
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
            return dt.isoformat()
        try:
            seconds = float(value)
        except ValueError:
            pass
        else:
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
            return dt.isoformat()

        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            raise ValueError
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError as exc:
        raise CommandError(
            "INVALID_TIMESTAMP",
            f"Timestamp '{value}' is not a valid UTC value.",
            hint="Provide ISO8601 (with timezone) or epoch seconds/milliseconds.",
        ) from exc


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]
) -> None:
    field_list = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=field_list)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise CommandError(
            "CONTRACT_NOT_FOUND",
            f"Contract CSV '{path}' is missing.",
            hint="Run validate before sessionize.",
        )

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CONTRACT_COLUMNS):
            raise CommandError(
                "INVALID_CONTRACT_HEADER",
                "Contract CSV must match the 9-column contract schema.",
            )
        rows: list[dict[str, str]] = []
        previous_timestamp: str | None = None
        for index, row in enumerate(reader, start=1):
            current_timestamp = row.get("timestamp_utc", "")
            if not current_timestamp:
                raise CommandError(
                    "MISSING_TIMESTAMP",
                    f"Row {index} is missing timestamp_utc.",
                    hint="Ensure contract.csv preserves the timestamp_utc column.",
                )
            if (
                previous_timestamp is not None
                and current_timestamp < previous_timestamp
            ):
                raise CommandError(
                    "REVERSED_ORDER",
                    (
                        "Contract CSV must be sorted by timestamp_utc; "
                        f"row {index} timestamp {current_timestamp} "
                        f"is earlier than previous {previous_timestamp}."
                    ),
                    hint=(
                        "Sort contract.csv ascending by timestamp_utc before running sessionize."
                    ),
                )
            previous_timestamp = current_timestamp
            rows.append(dict(row))

    return rows


def _load_sessioned_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise CommandError(
            "SESSIONED_NOT_FOUND",
            f"Sessioned CSV '{path}' is missing.",
            hint="Run sessionize before deltify.",
        )

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = SESSION_COLUMNS
        if reader.fieldnames != expected:
            raise CommandError(
                "INVALID_SESSIONED_HEADER",
                "Sessioned CSV must include contract columns plus delta_t/log_delta_t.",
            )
        rows: list[dict[str, str]] = []
        previous_timestamp: str | None = None
        for index, row in enumerate(reader, start=1):
            current_timestamp = row.get("timestamp_utc", "")
            if not current_timestamp:
                raise CommandError(
                    "MISSING_TIMESTAMP",
                    f"Row {index} is missing timestamp_utc.",
                    hint="Ensure sessioned.csv preserves the timestamp_utc column.",
                )
            if (
                previous_timestamp is not None
                and current_timestamp < previous_timestamp
            ):
                raise CommandError(
                    "REVERSED_ORDER",
                    (
                        "Sessioned CSV must be sorted by timestamp_utc; "
                        f"row {index} timestamp {current_timestamp} "
                        f"is earlier than previous {previous_timestamp}."
                    ),
                    hint=(
                        "Sort sessioned.csv ascending by timestamp_utc before running deltify."
                    ),
                )
            previous_timestamp = current_timestamp
            rows.append(dict(row))

    return rows


if __name__ == "__main__":  # pragma: no cover - CLI execution guard
    raise SystemExit(main())
