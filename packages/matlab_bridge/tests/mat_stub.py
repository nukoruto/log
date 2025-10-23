"""Local MATLAB MAT reader stub for tests."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from os import fspath
from pathlib import Path


@dataclass
class MatArray:
    """Minimal array-like container for MAT test data."""

    data: list[float]
    shape: tuple[int, int]

    def flatten(self) -> list[float]:
        """Return the stored data as a flat list."""

        return list(self.data)


def loadmat(path: str | bytes | bytearray | Path) -> dict[str, MatArray]:
    """Read a MATLAB v4 MAT file produced by the exporter."""

    raw_path = fspath(path)
    if isinstance(raw_path, bytes):
        raw_path = raw_path.decode()
    payload = Path(raw_path).read_bytes()
    offset = 0
    result: dict[str, MatArray] = {}

    while offset + 20 <= len(payload):
        _type_code, mrows, ncols, imagf, namelen = struct.unpack(
            "<5i", payload[offset : offset + 20]
        )
        offset += 20

        name_bytes = payload[offset : offset + namelen]
        offset += namelen
        name = name_bytes.rstrip(b"\x00").decode("ascii")

        count = mrows * ncols
        values = list(
            struct.unpack(
                "<" + "d" * count,
                payload[offset : offset + 8 * count],
            )
        )
        offset += 8 * count

        if imagf != 0:
            offset += 8 * count
            continue

        result[name] = MatArray(data=values, shape=(mrows, ncols))

    return result


__all__ = ["loadmat", "MatArray"]
