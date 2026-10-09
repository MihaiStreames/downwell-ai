class DownwellError(Exception):
    """Base error for all Downwell project errors."""


class FieldResolveError(DownwellError):
    """All pointer chains for a named game field failed."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"could not resolve field {field!r}: all pointer chains failed")


class GameTimeoutError(DownwellError):
    """The game stopped responding to the environment within its deadline."""

    def __init__(self, waiting_for: str, timeout: float) -> None:
        self.waiting_for = waiting_for
        super().__init__(f"timed out after {timeout:.0f}s waiting for {waiting_for}, check that downwell is running with the patched data.win")


class StateLayoutError(DownwellError):
    """The snapshot in the running game does not match `layout.py` (the patch is stale)."""

    def __init__(self, count: float, limit: int) -> None:
        self.count = count
        self.limit = limit
        super().__init__(f"snapshot reports {count} records but layout.py allows 0..{limit}, rerun scripts/gen_export.py and re-inject data.win")
