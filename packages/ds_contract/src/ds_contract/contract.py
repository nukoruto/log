"""契約CSVの列定義と検証ユーティリティ。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Final, Iterable, TextIO

CONTRACT_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp_utc",
    "uid",
    "session_id",
    "method",
    "path",
    "referer",
    "user_agent",
    "ip",
    "op_category",
)

REQUIRED_CONTRACT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "timestamp_utc",
        "uid",
        "session_id",
        "method",
        "path",
        "referer",
        "user_agent",
        "ip",
        "op_category",
    }
)

OPTIONAL_CONTRACT_FIELDS: Final[frozenset[str]] = frozenset(
    set(CONTRACT_COLUMNS) - set(REQUIRED_CONTRACT_FIELDS)
)


def ensure_required_fields(mapping: Iterable[str]) -> frozenset[str]:
    """Return the set of missing required fields for a given mapping iterable."""

    provided = set(mapping)
    missing = REQUIRED_CONTRACT_FIELDS.difference(provided)
    return frozenset(missing)


@dataclass(slots=True)
class StructuredLogger:
    """JSON構造ログを任意のストリームへ出力するロガー。"""

    command: str
    seed: int | None
    stream: TextIO = sys.stdout

    def emit(self, *, event: str, **payload: object) -> None:
        record: dict[str, object] = {
            "event": event,
            "command": self.command,
            "seed": self.seed,
        }
        record.update(payload)
        print(json.dumps(record, ensure_ascii=False), file=self.stream)
        try:
            self.stream.flush()
        except Exception:  # pragma: no cover - flush best effort
            pass

    def log_start(self, details: dict[str, object]) -> None:
        self.emit(event="start", details=details)

    def log_complete(self, details: dict[str, object]) -> None:
        self.emit(event="complete", details=details)

    def log_error(self, *, code: str, message: str, hint: str | None = None) -> None:
        payload: dict[str, object] = {"code": code, "message": message}
        if hint is not None:
            payload["hint"] = hint
        self.emit(event="error", **payload)


def validate_cli(
    input_csv: str | Path | None,
    *,
    mapping: str | Path | None,
    output: str | Path | None,
    meta: str | Path | None = None,
    seed: int | None,
    stream: TextIO = sys.stdout,
) -> int:
    """`ds-contract validate` 相当の処理を実行し JSON ログを返す。"""

    logger = StructuredLogger(command="validate", seed=seed, stream=stream)
    if seed is None:
        logger.log_error(
            code="MISSING_SEED",
            message="--seed is required for deterministic processing.",
            hint="Invoke the command with an explicit --seed integer value.",
        )
        return 1

    from . import cli as cli_module  # 遅延インポートで循環参照を回避

    cli_module._set_global_seed(seed)

    if input_csv is None:
        logger.log_error(
            code="ARGUMENT_ERROR",
            message="input_csv path is required.",
            hint="Pass the raw CSV path as the first positional argument (see --help).",
        )
        return 2

    if mapping is None:
        logger.log_error(
            code="ARGUMENT_ERROR",
            message="--map is required.",
            hint="Provide a YAML mapping via --map <path>. Run with --help for usage details.",
        )
        return 2

    if output is None:
        logger.log_error(
            code="ARGUMENT_ERROR",
            message="--out is required.",
            hint="Specify the contract CSV output path via --out <path> (see --help).",
        )
        return 2

    args = SimpleNamespace(
        command="validate",
        seed=seed,
        input_csv=str(Path(input_csv)),
        mapping=str(Path(mapping)),
        out=str(Path(output)),
        meta=str(Path(meta)) if meta is not None else None,
    )

    return _run_cli_handler("_handle_validate", args, logger)


def _run_cli_handler(
    handler_name: str, args: SimpleNamespace, logger: StructuredLogger
) -> int:
    """Invoke the CLI handler while統一したエラー処理で JSON ログを出力する。"""

    from . import cli  # 遅延インポートで循環参照を避ける

    handler = getattr(cli, handler_name)
    try:
        result = handler(args, logger)
    except cli.CommandError as exc:
        logger.log_error(code=exc.code, message=exc.message, hint=exc.hint)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - 予期しない障害
        logger.log_error(code="UNEXPECTED", message=str(exc))
        return 1
    logger.log_complete(result)
    return 0
