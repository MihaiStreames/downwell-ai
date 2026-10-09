def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    """Interpolate from `start` to `end` over `duration` steps, then hold at `end`."""
    progress = min(step / duration, 1.0)
    return start + progress * (end - start)
