import struct

import numpy as np
import pytest

from downwell_ai.env.memory._rvalue import KIND_ARRAY
from downwell_ai.env.memory._rvalue import KIND_BOOL
from downwell_ai.env.memory._rvalue import KIND_INT32
from downwell_ai.env.memory._rvalue import KIND_INT64
from downwell_ai.env.memory._rvalue import KIND_REAL
from downwell_ai.env.memory._rvalue import decode_rvalue
from downwell_ai.env.memory._rvalue import decode_rvalues


_KIND_UNDEFINED = 5


def _rvalue(kind: int, value: float) -> bytes:
    payload = struct.pack("<d", value) if kind == KIND_REAL else struct.pack("<q", int(value))
    return payload + struct.pack("<II", 0, kind)


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [(KIND_INT32, -7, -7.0), (KIND_INT64, 2**40, float(2**40)), (KIND_BOOL, 1, 1.0), (KIND_BOOL, 0, 0.0)],
)
def test_numeric_kinds_decode_to_their_value(kind: int, value: float, expected: float):
    assert decode_rvalue(_rvalue(kind, value)) == expected


def test_non_numeric_kind_decodes_to_nan():
    assert np.isnan(decode_rvalue(_rvalue(KIND_ARRAY, 0x1234)))


def test_kind_ignores_the_flag_bits_above_24():
    raw = struct.pack("<d", 2.0) + struct.pack("<II", 0, KIND_REAL | 0x80000000)

    assert decode_rvalue(raw) == 2.0


def test_packed_values_decode_per_kind_and_undefined_reads_as_zero():
    raw = b"".join([_rvalue(KIND_REAL, 1.25), _rvalue(KIND_INT32, 42), _rvalue(_KIND_UNDEFINED, 0), _rvalue(KIND_BOOL, 1), _rvalue(KIND_INT64, -3)])

    assert decode_rvalues(raw).tolist() == [1.25, 42.0, 0.0, 1.0, -3.0]
