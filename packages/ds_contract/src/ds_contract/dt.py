"""Δt ロバスト正規化モジュール。"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence, TextIO, TypedDict

from .contract import StructuredLogger, _run_cli_handler
from .sessionize import EPSILON, SESSION_COLUMNS, quantile


MAD_SCALE: float = 1.4826
CLIP_RANGE: tuple[float, float] = (-5.0, 5.0)

DELTIFIED_COLUMNS: list[str] = SESSION_COLUMNS + [
    "robust_z",
    "robust_z_clipped",
    "prev_q25_seconds",
    "prev_q75_seconds",
    "burst_ratio",
]


class UserStats(TypedDict):
    """ユーザ単位の統計情報。"""

    median: float
    mad: float


class DeltifyMeta(TypedDict):
    """deltify のメタ情報。"""

    epsilon: float
    mad_scale: float
    clip_range: tuple[float, float]
    global_median: float
    global_mad: float
    group: str
    fallback_users: list[str]
    user_stats: dict[str, UserStats]


@dataclass(slots=True)
class DeltifyResult:
    """deltify 処理の結果。"""

    rows: list[dict[str, str]]
    meta: DeltifyMeta


def deltify_session_rows(
    rows: Sequence[dict[str, str]] | Iterable[dict[str, str]],
) -> DeltifyResult:
    """セッション化済み行にロバスト Δt 特徴量を付与する。"""

    ordered_rows = [dict(row) for row in rows]
    if not ordered_rows:
        raise ValueError("Cannot deltify an empty sequence of rows.")

    ordered_rows.sort(key=lambda item: item["timestamp_utc"])

    log_values_per_user: dict[str, list[float]] = defaultdict(list)
    all_log_values: list[float] = []

    for row in ordered_rows:
        uid = row["uid"]
        delta_seconds = float(row["delta_t_seconds"])
        log_value = math.log(delta_seconds + EPSILON)
        log_values_per_user[uid].append(log_value)
        all_log_values.append(log_value)

    global_median = _median(all_log_values)
    global_mad = _mad(all_log_values, global_median)
    if math.isclose(global_mad, 0.0, abs_tol=1e-12):
        global_mad = 1e-9

    user_stats: dict[str, tuple[float, float]] = {}
    fallback_users: list[str] = []

    for uid, values in log_values_per_user.items():
        if len(values) >= 5:
            median = _median(values)
            mad = _mad(values, median)
            if math.isclose(mad, 0.0, abs_tol=1e-12):
                mad = global_mad
        else:
            median = global_median
            mad = global_mad
            fallback_users.append(uid)
        user_stats[uid] = (median, mad)

    fallback_users.sort()

    enriched_rows: list[dict[str, str]] = []
    previous_deltas: dict[str, list[float]] = defaultdict(list)

    for row in ordered_rows:
        uid = row["uid"]
        delta_seconds = float(row["delta_t_seconds"])
        log_delta = math.log(delta_seconds + EPSILON)
        median, mad = user_stats[uid]
        denominator = (
            MAD_SCALE * mad
            if not math.isclose(mad, 0.0, abs_tol=1e-12)
            else MAD_SCALE * global_mad
        )

        if math.isclose(delta_seconds, 0.0, abs_tol=1e-12):
            z_score = 0.0
        else:
            z_score = (log_delta - median) / denominator
        clipped = max(CLIP_RANGE[0], min(CLIP_RANGE[1], z_score))

        history = previous_deltas[uid]
        if history:
            q25 = float(quantile(history, 0.25))
            q75 = float(quantile(history, 0.75))
            prev_delta = history[-1]
        else:
            q25 = delta_seconds
            q75 = delta_seconds
            prev_delta = delta_seconds

        burst = (prev_delta + EPSILON) / (delta_seconds + EPSILON)

        enriched = dict(row)
        enriched["robust_z"] = f"{z_score:.6f}"
        enriched["robust_z_clipped"] = f"{clipped:.6f}"
        enriched["prev_q25_seconds"] = f"{q25:.6f}"
        enriched["prev_q75_seconds"] = f"{q75:.6f}"
        enriched["burst_ratio"] = f"{burst:.6f}"
        enriched_rows.append(enriched)

        history.append(delta_seconds)
        if len(history) > 5:
            history.pop(0)

    meta: DeltifyMeta = {
        "epsilon": EPSILON,
        "mad_scale": MAD_SCALE,
        "clip_range": CLIP_RANGE,
        "global_median": global_median,
        "global_mad": global_mad,
        "group": "uid",
        "fallback_users": fallback_users,
        "user_stats": {
            uid: {"median": stats[0], "mad": stats[1]}
            for uid, stats in sorted(user_stats.items())
        },
    }

    return DeltifyResult(rows=enriched_rows, meta=meta)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty values.")
    sorted_values = sorted(float(v) for v in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[mid])
    return float(sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _mad(values: Sequence[float], median: float) -> float:
    deviations = [abs(float(value) - median) for value in values]
    if not deviations:
        return 0.0
    return _median(deviations)


def deltify_cli(
    sessioned_csv: str | Path | None,
    *,
    output: str | Path | None,
    meta: str | Path | None,
    seed: int | None,
    stream: TextIO = sys.stdout,
) -> int:
    """`ds-contract deltify` を JSON ログ付きで実行するヘルパー。"""

    logger = StructuredLogger(command="deltify", seed=seed, stream=stream)
    if seed is None:
        logger.log_error(
            code="MISSING_SEED",
            message="--seed is required for deterministic processing.",
            hint="Invoke the command with an explicit --seed integer value.",
        )
        return 1

    from . import cli as cli_module  # 遅延インポートで循環参照を避ける

    cli_module._set_global_seed(seed)

    if sessioned_csv is None:
        logger.log_error(
            code="ARGUMENT_ERROR",
            message="sessioned_csv path is required.",
            hint="Provide the sessionized CSV path as input (see --help).",
        )
        return 2

    if output is None:
        logger.log_error(
            code="ARGUMENT_ERROR",
            message="--out is required.",
            hint="Specify the deltified CSV output path via --out <path> (see --help).",
        )
        return 2

    if meta is None:
        logger.log_error(
            code="ARGUMENT_ERROR",
            message="--meta is required.",
            hint="Provide the Δt metadata JSON path via --meta <path> (see --help).",
        )
        return 2

    args = SimpleNamespace(
        command="deltify",
        seed=seed,
        sessioned_csv=str(Path(sessioned_csv)),
        out=str(Path(output)),
        meta=str(Path(meta)),
    )

    return _run_cli_handler("_handle_deltify", args, logger)


__all__ = [
    "CLIP_RANGE",
    "DELTIFIED_COLUMNS",
    "DeltifyMeta",
    "DeltifyResult",
    "MAD_SCALE",
    "deltify_session_rows",
    "deltify_cli",
]
