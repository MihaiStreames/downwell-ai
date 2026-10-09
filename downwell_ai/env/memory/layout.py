from pathlib import Path
import tomllib
from typing import Final


_GAME_DATA: Final[dict] = tomllib.loads((Path(__file__).parent.parent / "downwell.toml").read_text(encoding="utf-8"))

HEADER_FIELDS: Final[tuple[str, ...]] = (
    "frame",
    "room",
    "has_player",
    "xpos",
    "ypos",
    "vx",
    "vy",
    "grounded",
    "inwater",
    "hp",
    "hp_max",
    "ammo",
    "ammo_max",
    "gems",
    "combo",
    "area",
    "level",
    "death",
    "in_side_room",
    "menu",
    "cursor",
    "shop",
    "shop_cursor",
    "view_x",
    "view_y",
    "view_w",
    "view_h",
)

RECORD_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "object",
    "flags",
    "x",
    "y",
    "left",
    "top",
    "right",
    "bottom",
    "vx",
    "vy",
    "hp",
    "yy",
    "white",
    "angry",
    "immune",
)

OFFER_SLOTS: Final[int] = 4
SHOP_SLOTS: Final[int] = 4
N_UPGRADES: Final[int] = len(_GAME_DATA["upgrades"]["names"])
MAX_RECORDS: Final[int] = 1024

FLOAT_BYTES: Final[int] = 4
PREFIX_FLOATS: Final[int] = len(HEADER_FIELDS) + OFFER_SLOTS + SHOP_SLOTS + N_UPGRADES + 1
RECORD_FLOATS: Final[int] = len(RECORD_FIELDS)
BUFFER_BYTES: Final[int] = (PREFIX_FLOATS + MAX_RECORDS * RECORD_FLOATS) * FLOAT_BYTES

HEADER_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(HEADER_FIELDS)}
RECORD_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(RECORD_FIELDS)}

# family bits in each record's `flags`
# computed once by the patch from each object's parent chain

FLAG_ENEMY: Final[int] = 1
FLAG_PLAYER_BULLET: Final[int] = 1 << 1
FLAG_SOLID: Final[int] = 1 << 2
FLAG_THIN: Final[int] = 1 << 3
FLAG_WATER: Final[int] = 1 << 4
FLAG_BREAKABLE: Final[int] = 1 << 5
FLAG_TIME_FIELD: Final[int] = 1 << 6
FLAG_LISTED: Final[int] = 1 << 7

# instances with any of these flags get their variables (vx, vy, hp, yy, white, angry, immune) exported
# the rest (walls, platforms, water) only need their position and box

DETAIL_FLAGS: Final[int] = FLAG_ENEMY | FLAG_PLAYER_BULLET | FLAG_LISTED

# flag -> objects whose instances (and whose descendants' instances) carry it
FAMILY_ROOTS: Final[dict[int, tuple[str, ...]]] = {
    FLAG_ENEMY: ("subparentEnemy",),
    FLAG_PLAYER_BULLET: ("parentPBullet",),
    FLAG_SOLID: ("sParentSolid",),
    FLAG_THIN: ("parentThinwall",),
    FLAG_WATER: ("parentWater",),
    FLAG_BREAKABLE: ("parentShootableThing", "subparentShootableWall"),
    FLAG_TIME_FIELD: ("parentTimeField",),
}

LISTED_OBJECTS: Final[tuple[str, ...]] = (
    *_GAME_DATA["hazards"],
    *_GAME_DATA["terrain_hazards"],
    *(name for name, pickup in _GAME_DATA["pickups"].items() if "parent" not in pickup),
    *_GAME_DATA["extras"],
)
