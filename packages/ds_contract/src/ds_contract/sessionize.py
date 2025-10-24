"""Sessionization utilities implementing Otsu and elbow selection."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence, TypedDict, cast

from .contract import CONTRACT_COLUMNS

EPSILON: float = 1e-3
SESSION_COLUMNS: list[str] = list(CONTRACT_COLUMNS) + [
    "delta_t_seconds",
    "log_delta_t",
]


class HistogramInfo(TypedDict):
    """Histogram configuration for session gap log values."""

    bin_edges: list[float]
    counts: list[int]
    fd_width: float | None
    bin_count: int
    min_log: float | None
    max_log: float | None


class BinsInfo(TypedDict):
    """Summary of histogram binning for metadata."""

    count: int
    fd_width: float | None
    min_log: float | None
    max_log: float | None


@dataclass(slots=True)
class SessionizationResult:
    """Result of sessionization including metadata."""

    rows: list[dict[str, str]]
    threshold_seconds: float
    method: str
    histogram: HistogramInfo
    bins: BinsInfo
    within_class_variance_ratio: float

    @property
    def log_threshold(self) -> float:
        """Return log threshold adjusted by epsilon."""

        return math.log(self.threshold_seconds + EPSILON)


def sessionize_contract(
    rows: Sequence[dict[str, str]] | Iterable[dict[str, str]],
) -> SessionizationResult:
    """Sessionize contract rows using Otsu or elbow threshold selection."""

    ordered_rows = [dict(row) for row in rows]
    if not ordered_rows:
        raise ValueError("Cannot sessionize an empty sequence of rows.")
    ordered_rows.sort(key=lambda item: item["timestamp_utc"])

    log_values, timestamps_per_uid = _collect_log_deltas(ordered_rows)

    if not log_values:
        histogram = cast(
            HistogramInfo,
            {
                "bin_edges": [],
                "counts": [],
                "fd_width": None,
                "bin_count": 0,
                "min_log": None,
                "max_log": None,
            },
        )
        bins_info = cast(
            BinsInfo,
            {
                "count": 0,
                "fd_width": None,
                "min_log": None,
                "max_log": None,
            },
        )
        threshold = 3600.0
        session_rows = _assign_sessions(ordered_rows, threshold)
        return SessionizationResult(
            rows=session_rows,
            threshold_seconds=threshold,
            method="elbow",
            histogram=histogram,
            bins=bins_info,
            within_class_variance_ratio=1.0,
        )

    histogram, fd_width = _build_histogram(log_values)
    ratio, otsu_threshold_log = _otsu_within_class_ratio(histogram)

    if ratio < 0.9:
        method = "otsu"
        threshold = math.exp(otsu_threshold_log)
    else:
        method = "elbow"
        threshold = _elbow_threshold(log_values, timestamps_per_uid)

    session_rows = _assign_sessions(ordered_rows, threshold)
    bins_info = cast(
        BinsInfo,
        {
            "count": histogram["bin_count"],
            "fd_width": fd_width,
            "min_log": histogram["min_log"],
            "max_log": histogram["max_log"],
        },
    )
    return SessionizationResult(
        rows=session_rows,
        threshold_seconds=threshold,
        method=method,
        histogram=histogram,
        bins=bins_info,
        within_class_variance_ratio=ratio,
    )


def quantile(values: Sequence[float], q: float) -> float:
    """Compute the q-quantile using linear interpolation."""

    if not values:
        raise ValueError("Cannot compute quantile of empty values.")
    sorted_values = sorted(values)
    return _quantile_sorted(sorted_values, q)


def _quantile_sorted(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty values.")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    weight = position - lower_index
    return lower_value * (1 - weight) + upper_value * weight


def _collect_log_deltas(
    ordered_rows: list[dict[str, str]],
) -> tuple[list[float], dict[str, list[datetime]]]:
    last_timestamp_per_uid: dict[str, datetime] = {}
    timestamps_per_uid: dict[str, list[datetime]] = defaultdict(list)
    log_values: list[float] = []

    for row in ordered_rows:
        uid = row["uid"]
        timestamp = datetime.fromisoformat(row["timestamp_utc"])
        previous = last_timestamp_per_uid.get(uid)
        if previous is not None:
            delta_seconds = max((timestamp - previous).total_seconds(), 0.0)
            if delta_seconds > 0:
                log_values.append(math.log(delta_seconds + EPSILON))
        last_timestamp_per_uid[uid] = timestamp
        timestamps_per_uid[uid].append(timestamp)

    return log_values, timestamps_per_uid


def _build_histogram(
    log_values: Sequence[float],
) -> tuple[HistogramInfo, float | None]:
    min_log = min(log_values)
    max_log = max(log_values)

    if math.isclose(max_log, min_log, rel_tol=1e-12, abs_tol=1e-12):
        histogram = cast(
            HistogramInfo,
            {
                "bin_edges": [float(min_log), float(max_log)],
                "counts": [len(log_values)],
                "fd_width": None,
                "bin_count": 1,
                "min_log": float(min_log),
                "max_log": float(max_log),
            },
        )
        return histogram, None

    q75 = quantile(list(log_values), 0.75)
    q25 = quantile(list(log_values), 0.25)
    iqr = q75 - q25
    count = len(log_values)
    width = 0.0
    if count > 0:
        width = 2 * iqr / (count ** (1 / 3)) if count > 0 else 0.0
    if width <= 0 or not math.isfinite(width):
        span = max_log - min_log
        width = span / 64 if span > 0 else 1.0

    span = max_log - min_log
    if width <= 0:
        bin_count = 32
    else:
        bin_count = int(math.ceil(span / width))
    bin_count = max(32, min(256, bin_count))

    counts, edges = _histogram(log_values, bin_count, min_log, max_log)
    histogram = cast(
        HistogramInfo,
        {
            "bin_edges": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
            "fd_width": float(width),
            "bin_count": bin_count,
            "min_log": float(min_log),
            "max_log": float(max_log),
        },
    )
    return histogram, float(width)


def _histogram(
    values: Sequence[float], bins: int, min_value: float, max_value: float
) -> tuple[list[int], list[float]]:
    if bins <= 0:
        bins = 1
    if math.isclose(max_value, min_value):
        return [len(values)], [float(min_value), float(max_value)]
    width = (max_value - min_value) / bins
    if width <= 0:
        width = 1.0
    edges = [min_value + i * width for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for value in values:
        if value >= max_value:
            index = bins - 1
        else:
            index = int((value - min_value) / width)
            index = max(0, min(index, bins - 1))
        counts[index] += 1
    return counts, edges


def _otsu_within_class_ratio(histogram: HistogramInfo) -> tuple[float, float]:
    counts = histogram["counts"]
    edges = histogram["bin_edges"]
    total = float(sum(counts))
    if total <= 0:
        return 1.0, float(edges[-1]) if edges else 0.0

    probabilities = [count / total for count in counts]
    bin_centers = [
        (edges[index] + edges[index + 1]) / 2 for index in range(len(counts))
    ]

    omega: list[float] = []
    mu: list[float] = []
    running_prob = 0.0
    running_mu = 0.0
    for prob, center in zip(probabilities, bin_centers):
        running_prob += prob
        running_mu += prob * center
        omega.append(running_prob)
        mu.append(running_mu)

    mu_total = mu[-1] if mu else 0.0
    sigma_between: list[float] = []
    for idx, w in enumerate(omega):
        if w <= 0 or w >= 1:
            sigma_between.append(0.0)
            continue
        numerator = (mu_total * w - mu[idx]) ** 2
        denominator = w * (1 - w)
        sigma_between.append(numerator / denominator if denominator else 0.0)

    if not sigma_between:
        return 1.0, bin_centers[-1] if bin_centers else 0.0

    otsu_index = max(range(len(sigma_between)), key=lambda i: sigma_between[i])
    otsu_threshold_log = bin_centers[otsu_index]

    w0 = omega[otsu_index]
    w1 = 1.0 - w0
    variance_total = sum(
        prob * (center - mu_total) ** 2
        for prob, center in zip(probabilities, bin_centers)
    )
    if w0 > 0:
        mu0 = mu[otsu_index] / w0
        var0 = (
            sum(
                probabilities[i] * (bin_centers[i] - mu0) ** 2
                for i in range(otsu_index + 1)
            )
            / w0
        )
    else:
        var0 = 0.0
    if w1 > 0:
        mu1 = (mu_total - mu[otsu_index]) / w1
        var1 = (
            sum(
                probabilities[i] * (bin_centers[i] - mu1) ** 2
                for i in range(otsu_index + 1, len(bin_centers))
            )
            / w1
        )
    else:
        var1 = 0.0
    if math.isclose(variance_total, 0.0):
        ratio = 1.0
    else:
        ratio = (w0 * var0 + w1 * var1) / variance_total

    return ratio, float(otsu_threshold_log)


def _elbow_threshold(
    log_values: Sequence[float], timestamps_per_uid: dict[str, list[datetime]]
) -> float:
    sorted_values = sorted(log_values)
    unique_values = sorted(set(sorted_values))
    if len(unique_values) == 1:
        return math.exp(unique_values[0])

    thresholds = unique_values
    min_val = thresholds[0]
    max_val = thresholds[-1]

    session_counts = []
    for tau in thresholds:
        threshold_seconds = math.exp(tau)
        count = _count_sessions_for_threshold(timestamps_per_uid, threshold_seconds)
        session_counts.append(count)

    range_val = max_val - min_val
    norm_x = [((tau - min_val) / range_val) if range_val else 0.0 for tau in thresholds]
    min_sessions = min(session_counts)
    max_sessions = max(session_counts)
    if max_sessions == min_sessions:
        return math.exp(sum(thresholds) / len(thresholds))
    session_range = max_sessions - min_sessions
    norm_y = [
        (count - min_sessions) / session_range if session_range else 0.0
        for count in session_counts
    ]

    x0 = norm_x[0]
    y0 = norm_y[0]
    x1 = norm_x[-1]
    y1 = norm_y[-1]
    slope = 0.0 if math.isclose(x1, x0) else (y1 - y0) / (x1 - x0)

    distances = [
        abs((ny - y0) - slope * (nx - x0)) / math.sqrt(1 + slope**2)
        for nx, ny in zip(norm_x, norm_y)
    ]
    best_index = max(range(len(distances)), key=lambda i: distances[i])
    return math.exp(thresholds[best_index])


def _count_sessions_for_threshold(
    timestamps_per_uid: dict[str, list[datetime]], threshold_seconds: float
) -> int:
    total_sessions = 0
    for times in timestamps_per_uid.values():
        if not times:
            continue
        count = 1
        for previous, current in zip(times, times[1:]):
            if (current - previous).total_seconds() > threshold_seconds:
                count += 1
        total_sessions += count
    return total_sessions


def _assign_sessions(
    rows: Sequence[dict[str, str]], threshold_seconds: float
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(dict(row))

    for user_rows in grouped.values():
        user_rows.sort(key=lambda item: item["timestamp_utc"])

    sessionized: list[dict[str, str]] = []
    for uid, user_rows in grouped.items():
        current_session = 0
        previous_time: datetime | None = None
        for row in user_rows:
            timestamp = datetime.fromisoformat(row["timestamp_utc"])
            if previous_time is None:
                delta_seconds = 0.0
                current_session += 1
            else:
                delta_seconds = max((timestamp - previous_time).total_seconds(), 0.0)
                if delta_seconds > threshold_seconds:
                    current_session += 1
            previous_time = timestamp

            session_row = dict(row)
            session_row["session_id"] = f"{uid}-{current_session:04d}"
            session_row["delta_t_seconds"] = f"{delta_seconds:.6f}"
            session_row["log_delta_t"] = f"{math.log(delta_seconds + EPSILON):.12f}"
            sessionized.append(session_row)

    sessionized.sort(key=lambda item: item["timestamp_utc"])
    return sessionized
