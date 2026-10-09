from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageSequence

from downwell_ai.utils.gif import save_gif


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_long_stall_saves_instead_of_overflowing_the_gif_delay(tmp_path: Path):
    path = tmp_path / "stall.gif"
    frames = [_frame(0), *[_frame(255)] * 12_000, _frame(0)]

    save_gif(frames, path, frame_skip=4)

    with Image.open(path) as gif:
        assert sum(1 for _ in ImageSequence.Iterator(gif)) == 3


def test_identical_frames_collapse_into_one_longer_frame(tmp_path: Path):
    path = tmp_path / "pause.gif"

    save_gif([_frame(0), _frame(255), _frame(255), _frame(0)], path, frame_skip=6)

    with Image.open(path) as gif:
        assert sum(1 for _ in ImageSequence.Iterator(gif)) == 3
        gif.seek(1)
        assert gif.info["duration"] == 200
