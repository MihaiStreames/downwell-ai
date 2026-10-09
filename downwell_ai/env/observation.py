from collections import deque
from enum import IntEnum
from typing import Final

from gymnasium.spaces import Box
from gymnasium.spaces import Dict
import numpy as np

from downwell_ai.env._game_data import DRONE_ID
from downwell_ai.env._game_data import GEM_ID
from downwell_ai.env._game_data import HAZARD_IDS
from downwell_ai.env._game_data import PICKUP_CATEGORY
from downwell_ai.env._game_data import TERRAIN_HAZARD_IDS
from downwell_ai.env._stomp import stomp_flags
from downwell_ai.env.memory import FLAG_BREAKABLE
from downwell_ai.env.memory import FLAG_ENEMY
from downwell_ai.env.memory import FLAG_PLAYER_BULLET
from downwell_ai.env.memory import FLAG_SOLID
from downwell_ai.env.memory import FLAG_THIN
from downwell_ai.env.memory import FLAG_TIME_FIELD
from downwell_ai.env.memory import FLAG_WATER
from downwell_ai.env.memory import N_UPGRADES
from downwell_ai.env.memory import OFFER_SLOTS
from downwell_ai.env.memory import RECORD_INDEX
from downwell_ai.env.memory import SHOP_SLOTS
from downwell_ai.env.memory import MemoryState


class Channel(IntEnum):
    """Grid channels, each a 0/1 map of what covers a cell of the game's view."""

    SOLID = 0
    THIN = 1
    SPIKES = 2
    STOMP_NOW = 3
    STOMP_FROM_ABOVE = 4
    HURTS = 5
    HAZARD = 6
    GEM = 7
    PICKUP = 8
    OWN_SHOT = 9
    PLAYER = 10
    WATER = 11
    SIDE_ROOM = 12


# what the game shows in rmMain
# the grid covers exactly that view, so it truncates like the screen

VIEW_W: Final[int] = 160
VIEW_H: Final[int] = 284

GRID_CELL_PX: Final[int] = 8
GRID_W: Final[int] = VIEW_W // GRID_CELL_PX
GRID_H: Final[int] = -(-VIEW_H // GRID_CELL_PX)

FRAME_SHAPE: Final[tuple[int, int, int]] = (len(Channel), GRID_H, GRID_W)

MOTION_CHANNELS: Final[tuple[Channel, ...]] = (Channel.STOMP_NOW, Channel.STOMP_FROM_ABOVE, Channel.HURTS, Channel.HAZARD)
MOTION_HISTORY: Final[int] = 2

GRID_SHAPE: Final[tuple[int, int, int]] = (len(Channel) + MOTION_HISTORY * len(MOTION_CHANNELS), GRID_H, GRID_W)

_PLAYER_HEIGHT_PX: Final[float] = 16.0


def _section(start: int, size: int) -> slice:
    return slice(start, start + size)


# each section starts where the previous one ends
# adding a feature can never shift another reader out of sync

_VEC_PLAYER: Final[slice] = _section(0, 6)
_VEC_COMBAT: Final[slice] = _section(_VEC_PLAYER.stop, 5)
_VEC_PROGRESS: Final[slice] = _section(_VEC_COMBAT.stop, 4)

# menu flag, cursor one-hot, then one upgrade one-hot per offer slot
VEC_MENU: Final[slice] = _section(_VEC_PROGRESS.stop, 1 + OFFER_SLOTS + OFFER_SLOTS * N_UPGRADES)
# shop flag, cursor one-hot, prices
VEC_SHOP: Final[slice] = _section(VEC_MENU.stop, 1 + SHOP_SLOTS + SHOP_SLOTS)

_VEC_OWNED: Final[slice] = _section(VEC_SHOP.stop, N_UPGRADES)
VEC_DIM: Final[int] = _VEC_OWNED.stop

# scaling keeps vector entries roughly in [-1, 1]
_VELOCITY_SCALE: Final[float] = 10.0
_GEMS_SCALE: Final[float] = 1000.0
_PRICE_SCALE: Final[float] = 1000.0
_LEVEL_SCALE: Final[float] = 16.0

OBS_SPACE = Dict(
    {
        "grid": Box(0, 1, GRID_SHAPE, dtype=np.uint8),
        "vec": Box(-np.inf, np.inf, (VEC_DIM,), dtype=np.float32),
    }
)


_NOT_DRAWN: Final[int] = -1

_TERRAIN_HAZARD: Final[int] = 1
_HAZARD: Final[int] = 2
_PICKUP: Final[int] = 4

# lookup table since this runs for every instance on every step
_KIND_BY_OBJECT: Final[np.ndarray] = np.zeros(max(*TERRAIN_HAZARD_IDS, *HAZARD_IDS, *PICKUP_CATEGORY) + 1, dtype=np.uint8)
for _ids, _kind in ((TERRAIN_HAZARD_IDS, _TERRAIN_HAZARD), (HAZARD_IDS, _HAZARD), (PICKUP_CATEGORY, _PICKUP)):
    _KIND_BY_OBJECT[list(_ids)] |= _kind


def _channels(flags: np.ndarray, objects: np.ndarray, stomp_now: np.ndarray, stomp_from_above: np.ndarray) -> np.ndarray:
    """Channel of each instance row, `_NOT_DRAWN` for rows no channel covers."""
    is_known = (objects >= 0) & (objects < len(_KIND_BY_OBJECT))
    kinds = np.where(is_known, _KIND_BY_OBJECT[np.where(is_known, objects, 0)], 0)
    is_terrain_hazard = kinds & _TERRAIN_HAZARD != 0
    enemy_channels = np.where(stomp_now, Channel.STOMP_NOW.value, np.where(stomp_from_above, Channel.STOMP_FROM_ABOVE.value, Channel.HURTS.value))

    # most specific first, the first match wins
    rules = (
        ((flags & FLAG_ENEMY != 0) & ~is_terrain_hazard, enemy_channels),
        (is_terrain_hazard, Channel.SPIKES),
        (kinds & _HAZARD != 0, Channel.HAZARD),
        (objects == GEM_ID, Channel.GEM),
        (flags & FLAG_TIME_FIELD != 0, Channel.SIDE_ROOM),
        (kinds & _PICKUP != 0, Channel.PICKUP),
        ((flags & FLAG_PLAYER_BULLET != 0) | (objects == DRONE_ID), Channel.OWN_SHOT),
        (flags & FLAG_SOLID != 0, Channel.SOLID),
        (flags & FLAG_BREAKABLE != 0, Channel.PICKUP),
        (flags & FLAG_THIN != 0, Channel.THIN),
        (flags & FLAG_WATER != 0, Channel.WATER),
    )

    # applied least specific first so each more specific rule overwrites it
    channels = np.full(len(flags), _NOT_DRAWN, dtype=np.int64)
    for matches, channel in reversed(rules):
        channels[matches] = channel[matches] if isinstance(channel, np.ndarray) else channel.value

    return channels


def _paint(grid: np.ndarray, channel: Channel, box: tuple[float, float, float, float], origin: tuple[float, float]) -> None:
    left, top, right, bottom = box
    view_x, view_y = origin

    first_column = max(int((left - view_x) // GRID_CELL_PX), 0)
    last_column = min(int((right - view_x) // GRID_CELL_PX), GRID_W - 1)
    first_row = max(int((top - view_y) // GRID_CELL_PX), 0)
    last_row = min(int((bottom - view_y) // GRID_CELL_PX), GRID_H - 1)

    if first_column <= last_column and first_row <= last_row:
        grid[channel, first_row : last_row + 1, first_column : last_column + 1] = 1


def encode_grid(state: MemoryState) -> np.ndarray:
    """Rasterize every kept instance's collision box over the game's view into one `FRAME_SHAPE` frame."""
    grid = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    view_x, view_y, view_w, _ = state.view
    if view_w <= 0:
        return grid

    rows = state.instances

    stomp_now, stomp_from_above = stomp_flags(state)
    flags = rows[:, RECORD_INDEX["flags"]].astype(np.int32)
    objects = rows[:, RECORD_INDEX["object"]].astype(np.int32)
    boxes = rows[:, [RECORD_INDEX["left"], RECORD_INDEX["top"], RECORD_INDEX["right"], RECORD_INDEX["bottom"]]]

    channels = _channels(flags, objects, stomp_now, stomp_from_above)

    # computed for every row at once since this runs on every step
    columns = ((boxes[:, [0, 2]] - view_x) // GRID_CELL_PX).astype(np.int32)
    lines = ((boxes[:, [1, 3]] - view_y) // GRID_CELL_PX).astype(np.int32)
    first_columns, last_columns = np.maximum(columns[:, 0], 0), np.minimum(columns[:, 1], GRID_W - 1)
    first_rows, last_rows = np.maximum(lines[:, 0], 0), np.minimum(lines[:, 1], GRID_H - 1)

    is_drawn = (channels != _NOT_DRAWN) & (first_columns <= last_columns) & (first_rows <= last_rows)
    for channel, first_row, last_row, first_column, last_column in zip(
        channels[is_drawn].tolist(),
        first_rows[is_drawn].tolist(),
        last_rows[is_drawn].tolist(),
        first_columns[is_drawn].tolist(),
        last_columns[is_drawn].tolist(),
        strict=True,
    ):
        grid[channel, first_row : last_row + 1, first_column : last_column + 1] = 1

    if state.has_player:
        player_box = (state.xpos, state.ypos - _PLAYER_HEIGHT_PX, state.xpos, state.ypos)
        _paint(grid, Channel.PLAYER, player_box, (view_x, view_y))

    return grid


def encode_vector(state: MemoryState) -> np.ndarray:
    """Flatten the non-spatial fields into a `VEC_DIM` vector."""
    hp_max = max(state.hp_max, 1)
    ammo_max = max(state.ammo_max, 1)
    view_x, view_y, view_w, view_h = state.view
    in_view_x = (state.xpos - view_x) / view_w if view_w > 0 else 0.0
    in_view_y = (state.ypos - view_y) / view_h if view_h > 0 else 0.0

    vec = np.zeros(VEC_DIM, dtype=np.float32)
    # assigning per section raises if a section's contents and its declared size ever disagree
    vec[_VEC_PLAYER] = (state.vx / _VELOCITY_SCALE, state.vy / _VELOCITY_SCALE, state.grounded, state.inwater, in_view_x, in_view_y)
    vec[_VEC_COMBAT] = (state.hp / hp_max, hp_max / 10, state.ammo / ammo_max, ammo_max / 10, state.combo / 10)
    vec[_VEC_PROGRESS] = (state.gems / _GEMS_SCALE, state.area / 4, state.level / _LEVEL_SCALE, state.in_side_room)

    if state.menu:
        menu = vec[VEC_MENU]
        menu[0] = 1

        if 0 <= state.cursor < OFFER_SLOTS:
            menu[1 + state.cursor] = 1

        offers = menu[1 + OFFER_SLOTS :].reshape(OFFER_SLOTS, N_UPGRADES)

        for slot, upgrade in enumerate(state.offers[:OFFER_SLOTS]):
            # unfilled offers read -1 and a failed roll leaves the game's placeholder 100
            if 0 <= upgrade < N_UPGRADES:
                offers[slot, upgrade] = 1

    if state.shop:
        shop = vec[VEC_SHOP]
        shop[0] = 1

        if 0 <= state.shop_cursor < SHOP_SLOTS:
            shop[1 + state.shop_cursor] = 1

        for slot, price in enumerate(state.shop_prices[:SHOP_SLOTS]):
            shop[1 + SHOP_SLOTS + slot] = price / _PRICE_SCALE

    vec[_VEC_OWNED][: len(state.owned)] = np.minimum(state.owned[:N_UPGRADES], 1)
    return vec


class ObservationEncoder:
    """Build observations matching `OBS_SPACE`, with the moving channels of the last `MOTION_HISTORY` steps after the current frame."""

    def __init__(self) -> None:
        self.__history: deque[np.ndarray] = deque(maxlen=MOTION_HISTORY)
        self.__place: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        """Forget the past frames, for a new episode."""
        self.__history.clear()
        self.__place = None

    def encode(self, state: MemoryState) -> dict[str, np.ndarray]:
        """Encode `state` on top of the frames seen since the view last jumped."""
        place = (state.area, state.level, state.room, state.in_side_room)
        if place != self.__place:
            self.__history.clear()
            self.__place = place

        frame = encode_grid(state)
        motion = frame[list(MOTION_CHANNELS)]

        past = [*self.__history, *[motion] * (MOTION_HISTORY - len(self.__history))]
        self.__history.appendleft(motion)

        return {"grid": np.concatenate([frame, *past]), "vec": encode_vector(state)}
