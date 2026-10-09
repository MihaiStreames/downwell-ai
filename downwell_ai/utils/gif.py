from pathlib import Path
from typing import Final

from loguru import logger
import numpy as np
from PIL import Image

from downwell_ai.consts import GAME_FPS


_GIF_MAX_FRAME_MS: Final[int] = 655_350  # delay is stored in centiseconds, 16 bits


def save_gif(frames: list[np.ndarray], path: Path, frame_skip: int) -> None:
    """
    Write `frames` as a looping gif played back at game speed, each frame lasting `frame_skip` game frames.

    NOTE: Identical consecutive frames collapse into one longer frame, capped at the format's per-frame delay limit.
    """
    if not frames:
        logger.warning("no frames to save, skipping {}", path.name)
        return

    # play back at game speed so stalling looks as long as it was, up to the gif's per-frame delay limit
    frame_ms = round(1000 * frame_skip / GAME_FPS)
    unique_frames: list[np.ndarray] = []
    durations: list[int] = []

    # collapse identical frames ourselves, pillow would merge them past the 16-bit delay field
    for frame in frames:
        if unique_frames and np.array_equal(frame, unique_frames[-1]):
            durations[-1] = min(durations[-1] + frame_ms, _GIF_MAX_FRAME_MS)
            continue

        unique_frames.append(frame)
        durations.append(frame_ms)

    images = [Image.fromarray(frame) for frame in unique_frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
