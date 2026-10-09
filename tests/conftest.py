import dataclasses
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from downwell_ai.config import RewardConfig
from downwell_ai.config import TrainingConfig
from downwell_ai.env.memory import RECORD_FIELDS
from downwell_ai.env.memory import RECORD_INDEX
from downwell_ai.env.memory import MemoryState
from downwell_ai.env.observation import GRID_SHAPE
from downwell_ai.env.observation import VEC_DIM


_rng = np.random.default_rng()  # new way of handling random number generation
_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.toml"


def rand_grid() -> np.ndarray:
    return _rng.integers(0, 2, GRID_SHAPE, dtype=np.uint8)


def rand_vec() -> np.ndarray:
    return _rng.random(VEC_DIM).astype(np.float32)


def make_state(**overrides: Any) -> MemoryState:  # noqa: ANN401 (any MemoryState field)
    """Alive player mid-level, with `overrides` applied."""
    base = MemoryState(
        xpos=100.0,
        ypos=0.0,
        vx=0.0,
        vy=0.0,
        grounded=0,
        inwater=0,
        hp=4,
        hp_max=4,
        ammo=8,
        ammo_max=8,
        gems=0,
        combo=0,
        area=1,
        level=1,
        death=0,
        frame=0,
        room=7,
        has_player=1,
        view=(0.0, 0.0, 160.0, 284.0),
    )
    return dataclasses.replace(base, **overrides)


def make_row(obj: int, flags: int, box: tuple[float, float, float, float], **fields: float) -> np.ndarray:
    """One instance record (see `layout.RECORD_FIELDS`) for object `obj` covering `box` (left, top, right, bottom)."""
    row = np.zeros(len(RECORD_FIELDS), dtype=np.float32)
    left, top, right, bottom = box
    values = {"object": obj, "flags": flags, "x": left, "y": bottom, "yy": bottom, "left": left, "top": top, "right": right, "bottom": bottom}
    for name, value in {**values, **fields}.items():
        row[RECORD_INDEX[name]] = value
    return row


def make_rows(*rows: np.ndarray) -> np.ndarray:
    """Stack records into a `MemoryState.instances` array."""
    return np.stack(rows) if rows else np.zeros((0, len(RECORD_FIELDS)), dtype=np.float32)


def make_config(**overrides: Any) -> TrainingConfig:  # noqa: ANN401 (any TrainingConfig field)
    """`configs/example.toml` with `overrides` applied; tests pin every value they assert on."""
    config = msgspec.toml.decode(_DEFAULT_CONFIG.read_bytes(), type=TrainingConfig)
    return msgspec.structs.replace(config, **overrides)


def make_reward_config(**overrides: Any) -> RewardConfig:  # noqa: ANN401 (any RewardConfig field)
    """Return the default reward config with `overrides` applied."""
    return msgspec.structs.replace(make_config().reward, **overrides)
