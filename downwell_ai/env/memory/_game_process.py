from dataclasses import dataclass
from dataclasses import field
import math
import struct

from loguru import logger
from PyMemoryEditor import OpenProcess
from PyMemoryEditor.process.abstract import AbstractProcess
from PyMemoryEditor.process.errors import ProcessNotFoundError

from downwell_ai.consts import PROCESS_NAME
from downwell_ai.utils.exceptions import FieldResolveError
from downwell_ai.utils.exceptions import StateLayoutError

from ._module_base import resolve_module_base
from ._rvalue import KIND_REAL
from ._rvalue import RVALUE_KIND_OFFSET
from ._rvalue import RVALUE_SIZE
from ._rvalue import decode_rvalue
from .generated.game_ptrs import FIELD_OFFSETS
from .generated.game_ptrs import GLOBALS_BASE
from .generated.game_ptrs import MAP_CHAIN
from .layout import FLOAT_BYTES
from .layout import MAX_RECORDS
from .layout import PREFIX_FLOATS
from .layout import RECORD_FLOATS


@dataclass(frozen=True)
class GameProcess:
    """
    Attached Downwell process, reading and writing the patch's globals through `generated/game_ptrs.py`.

    Invariant: both `_process` and `_module_base` are guaranteed valid for the lifetime of this object.
    Do not use standalone; use `open_game_process()`.
    """

    _process: AbstractProcess
    _module_base: int
    _addresses: dict[str, int] = field(default_factory=dict)

    def _read_ptr(self, addr: int) -> int | None:
        # PyMemoryEditor does not raise on bad reads, returns garbage on failure
        data: bytes = self._process.read_process_memory(addr, bytes, 4)
        result = struct.unpack_from("<I", data)[0]
        logger.trace(f"reading ptr {result:#x}")
        return result if result != 0 else None

    def address(self, field: str) -> int:
        """Address of `field`'s RValue. Raises `FieldResolveError` if its pointer chain breaks."""
        if field in self._addresses:
            return self._addresses[field]

        addr = self._read_ptr(self._module_base + GLOBALS_BASE)
        for offset in (*MAP_CHAIN, FIELD_OFFSETS[field]):
            if addr is None:
                break

            addr = self._read_ptr(addr + offset)

        if addr is None:
            raise FieldResolveError(field)

        self._addresses[field] = addr
        return addr

    def read_rvalue(self, addr: int) -> float:
        """Numeric value of the RValue at `addr`, cheap enough to poll."""
        raw: bytes = self._process.read_process_memory(addr, bytes, RVALUE_SIZE)
        logger.trace(f"reading rvalue at {addr:#x}")
        return decode_rvalue(raw)

    def read_field(self, field: str) -> float:
        """Numeric value of the global behind `field`."""
        return self.read_rvalue(self.address(field))

    def write_field(self, field: str, value: float) -> None:
        """Store `value` as a REAL in the global behind `field`."""
        addr = self.address(field)
        self._process.write_process_memory(addr, float, 8, float(value))
        # force REAL, an INT32 rvalue would reinterpret our double as garbage
        self._process.write_process_memory(addr + RVALUE_KIND_OFFSET, int, 4, KIND_REAL)

    def read_snapshot(self) -> bytes:
        """
        Read the packed snapshot the patch wrote, as laid out in `layout.py`: the fixed prefix, then its records.

        Raises `FieldResolveError` if the patch is missing and `StateLayoutError` if the record count is impossible (stale patch).
        """
        address = self.read_field("buf_addr")
        if not math.isfinite(address) or address <= 0:
            field_name = "buf_addr"
            raise FieldResolveError(field_name)

        base = int(address)
        prefix_bytes = PREFIX_FLOATS * FLOAT_BYTES
        prefix: bytes = self._process.read_process_memory(base, bytes, prefix_bytes)

        count = struct.unpack_from("<f", prefix, prefix_bytes - FLOAT_BYTES)[0]
        if not 0 <= count <= MAX_RECORDS or count != int(count):
            raise StateLayoutError(count, MAX_RECORDS)

        record_bytes = int(count) * RECORD_FLOATS * FLOAT_BYTES
        logger.trace(f"reading snapshot at {base:#x}: {int(count)} records")
        if record_bytes == 0:
            return prefix

        return prefix + self._process.read_process_memory(base + prefix_bytes, bytes, record_bytes)

    def close(self) -> None:
        """Terminate session. This object must not be used after calling this."""
        logger.debug(f"closed {self._process._process_info.name}")  # noqa: SLF001 (readability)
        self._process.close()


def open_game_process(proc_name: str = PROCESS_NAME) -> GameProcess | None:
    """Attempt to find and attach to a process. Works on both Windows (via `OpenProcess`) and Linux (via `/proc/`)."""
    try:
        proc = OpenProcess(name=proc_name)
    except ProcessNotFoundError:
        return None

    base = resolve_module_base(proc, proc_name)
    if base is None:
        logger.error(f"attached to {proc_name} but module base not found in memory")
        proc.close()
        return None

    logger.debug(f"attached to {proc_name} (base 0x{base:x})")
    return GameProcess(proc, base)
