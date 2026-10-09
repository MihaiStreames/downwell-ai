from typing import Final

import numpy as np

from downwell_ai.env.memory import N_UPGRADES
from downwell_ai.env.memory import OFFER_SLOTS
from downwell_ai.env.observation import GRID_H
from downwell_ai.env.observation import GRID_W
from downwell_ai.env.observation import VEC_MENU
from downwell_ai.env.observation import Channel


_CELL_PX: Final = 8

_BACKGROUND: Final = (16, 16, 24)
_CURSOR: Final = (255, 255, 255)

# drawn in this order, so later channels sit on top (the player last)
_CHANNEL_COLORS: Final[dict[Channel, tuple[int, int, int]]] = {
    Channel.WATER: (30, 50, 110),
    Channel.SIDE_ROOM: (60, 40, 80),
    Channel.SOLID: (120, 120, 130),
    Channel.THIN: (150, 100, 50),
    Channel.SPIKES: (200, 200, 40),
    Channel.PICKUP: (80, 200, 255),
    Channel.GEM: (255, 60, 200),
    Channel.HAZARD: (255, 160, 0),
    Channel.HURTS: (220, 40, 40),
    Channel.STOMP_FROM_ABOVE: (40, 140, 70),
    Channel.STOMP_NOW: (90, 255, 120),
    Channel.OWN_SHOT: (255, 255, 160),
    Channel.PLAYER: (255, 255, 255),
}


def _upgrade_color(upgrade: int) -> tuple[int, int, int]:
    # spread the ids around the hue circle so different offers read as different colors
    hue = upgrade / N_UPGRADES * 6
    sector, fraction = int(hue), hue - int(hue)
    rising, falling = int(255 * fraction), int(255 * (1 - fraction))
    return [(255, rising, 0), (falling, 255, 0), (0, 255, rising), (0, falling, 255), (rising, 0, 255), (255, 0, falling)][sector % 6]


def _render_menu(menu: np.ndarray) -> np.ndarray:
    frame = np.empty((GRID_H * _CELL_PX, GRID_W * _CELL_PX, 3), dtype=np.uint8)
    frame[:] = _BACKGROUND

    cursor = menu[1 : 1 + OFFER_SLOTS]
    offers = menu[1 + OFFER_SLOTS :].reshape(OFFER_SLOTS, N_UPGRADES)

    box = frame.shape[1] // OFFER_SLOTS
    top = frame.shape[0] // 2 - box // 2

    for slot in range(OFFER_SLOTS):
        left = slot * box

        if cursor[slot]:
            frame[top - 4 : top + box + 4, left : left + box] = _CURSOR
        if offers[slot].any():
            frame[top : top + box, left + 4 : left + box - 4] = _upgrade_color(int(offers[slot].argmax()))

    return frame


def render_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Draw what the agent sees as an RGB frame, one color per grid channel over the game's view.

    Built from the observation rather than the screen, so a gif shows exactly the information the policy acted on.
    In the upgrade menu it draws the offers instead, one color per upgrade, with the cursor's slot outlined.
    """
    grid, vec = obs["grid"], obs["vec"]
    if vec[VEC_MENU][0]:
        return _render_menu(vec[VEC_MENU])

    frame = np.empty((GRID_H, GRID_W, 3), dtype=np.uint8)
    frame[:] = _BACKGROUND
    for channel, color in _CHANNEL_COLORS.items():
        frame[grid[channel] == 1] = color

    return frame.repeat(_CELL_PX, axis=0).repeat(_CELL_PX, axis=1)
