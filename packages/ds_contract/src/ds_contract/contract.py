"""契約CSVの列定義と検証ユーティリティ。"""

from __future__ import annotations

from typing import Final, Iterable

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
