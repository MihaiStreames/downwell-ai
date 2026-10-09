from dataclasses import dataclass
from dataclasses import field

import numpy as np

from .layout import HEADER_FIELDS
from .layout import HEADER_INDEX
from .layout import N_UPGRADES
from .layout import OFFER_SLOTS
from .layout import RECORD_FLOATS
from .layout import SHOP_SLOTS


def _no_instances() -> np.ndarray:
    return np.zeros((0, RECORD_FLOATS), dtype=np.float32)


@dataclass
class MemoryState:
    """One exported snapshot: the player and run state, plus every active instance the agent cares about."""

    xpos: float
    ypos: float
    vx: float
    vy: float
    grounded: int
    inwater: int
    hp: int
    hp_max: int
    ammo: int
    ammo_max: int
    gems: int
    combo: int
    area: int
    level: int
    death: int
    frame: int
    room: int
    has_player: int
    in_side_room: int = 0
    menu: int = 0
    cursor: int = 0
    shop: int = 0
    shop_cursor: int = 0
    view: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    offers: list[int] = field(default_factory=list)
    shop_prices: list[int] = field(default_factory=list)
    owned: list[int] = field(default_factory=list)
    instances: np.ndarray = field(default_factory=_no_instances)


def parse_snapshot(raw: bytes) -> MemoryState:
    """Build a `MemoryState` from the patch's snapshot buffer, ordered as in `layout.py`."""
    values = np.frombuffer(raw, dtype="<f4")
    header = values[: len(HEADER_FIELDS)]

    def get(name: str) -> float:
        return float(header[HEADER_INDEX[name]])

    cursor = len(HEADER_FIELDS)
    offers = values[cursor : cursor + OFFER_SLOTS]
    cursor += OFFER_SLOTS
    prices = values[cursor : cursor + SHOP_SLOTS]
    cursor += SHOP_SLOTS
    owned = values[cursor : cursor + N_UPGRADES]
    cursor += N_UPGRADES
    count = int(values[cursor])
    cursor += 1
    instances = values[cursor : cursor + count * RECORD_FLOATS].reshape(count, RECORD_FLOATS).copy()

    return MemoryState(
        xpos=get("xpos"),
        ypos=get("ypos"),
        vx=get("vx"),
        vy=get("vy"),
        grounded=int(get("grounded")),
        inwater=int(get("inwater")),
        hp=int(get("hp")),
        hp_max=int(get("hp_max")),
        ammo=int(get("ammo")),
        ammo_max=int(get("ammo_max")),
        gems=int(get("gems")),
        combo=int(get("combo")),
        area=int(get("area")),
        level=int(get("level")),
        death=int(get("death")),
        frame=int(get("frame")),
        room=int(get("room")),
        has_player=int(get("has_player")),
        in_side_room=int(get("in_side_room")),
        menu=int(get("menu")),
        cursor=int(get("cursor")),
        shop=int(get("shop")),
        shop_cursor=int(get("shop_cursor")),
        view=(get("view_x"), get("view_y"), get("view_w"), get("view_h")),
        offers=[int(offer) for offer in offers],
        shop_prices=[int(price) for price in prices],
        owned=[int(amount) for amount in owned],
        instances=instances,
    )
