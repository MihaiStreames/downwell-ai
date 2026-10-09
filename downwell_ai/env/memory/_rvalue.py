import struct
from typing import Final

import numpy as np


# gamemaker RValue: 8-byte value, 4 bytes flags, 4 bytes kind
RVALUE_SIZE: Final[int] = 16
RVALUE_KIND_OFFSET: Final[int] = 12
_RVALUE_KIND_MASK: Final[int] = 0xFFFFFF

KIND_REAL: Final[int] = 0
KIND_ARRAY: Final[int] = 2
KIND_PTR: Final[int] = 3
KIND_INT32: Final[int] = 7
KIND_INT64: Final[int] = 10
KIND_BOOL: Final[int] = 13

# RefDynamicArrayOfRValue, gamemaker 2022.5 x86 runner
ARRAY_ELEMENTS_OFFSET: Final[int] = 0x64
ARRAY_LENGTH_OFFSET: Final[int] = 0x74


def rvalue_kind(raw: bytes) -> int:
    """Kind tag of one raw RValue."""
    return struct.unpack_from("<I", raw, RVALUE_KIND_OFFSET)[0] & _RVALUE_KIND_MASK


def decode_rvalue(raw: bytes) -> float:
    """Numeric value of one raw RValue; kinds that are not numbers read as nan."""
    kind = rvalue_kind(raw)

    # integer literals in gml compile to INT32 rvalues, everything computed is REAL
    if kind == KIND_REAL:
        return struct.unpack_from("<d", raw)[0]
    if kind == KIND_INT32:
        return float(struct.unpack_from("<i", raw)[0])
    if kind == KIND_INT64:
        return float(struct.unpack_from("<q", raw)[0])
    if kind == KIND_BOOL:
        return 1.0 if struct.unpack_from("<Q", raw)[0] else 0.0
    # pointers (buffer_get_address) are 32-bit addresses on this x86 runner
    if kind == KIND_PTR:
        return float(struct.unpack_from("<I", raw)[0])

    return float("nan")


def decode_rvalues(raw: bytes) -> np.ndarray:
    """Numeric values of consecutive raw RValues; undefined and other non-number kinds read as 0."""
    count = len(raw) // RVALUE_SIZE

    # one row per RValue: [low word, high word, flags, kind]
    words = np.frombuffer(raw, dtype="<i4").reshape(count, 4)
    reals = np.frombuffer(raw, dtype="<f8").reshape(count, 2)[:, 0]
    longs = np.frombuffer(raw, dtype="<i8").reshape(count, 2)[:, 0]
    kinds = words[:, 3] & _RVALUE_KIND_MASK

    values = np.zeros(count)
    values = np.where(kinds == KIND_REAL, reals, values)
    values = np.where(kinds == KIND_INT32, words[:, 0], values)
    values = np.where(kinds == KIND_INT64, longs, values)

    return np.where(kinds == KIND_BOOL, longs != 0, values)
