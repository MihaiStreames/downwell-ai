from enum import StrEnum
from pathlib import Path
from typing import Final

import msgspec
from msgspec import Struct


class Stomp(StrEnum):
    """When landing on an enemy stomps it, from its player-collision handler in the game code."""

    YES = "yes"
    FALLING = "falling"
    WHEN_WHITE = "when_white"
    WHEN_CALM = "when_calm"
    NO = "no"


class PickupCategory(StrEnum):
    """What a pickup offers the player."""

    HEALTH = "health"
    CHARGE = "charge"
    MODULE = "module"
    GEM_SOURCE = "gem_source"
    SIDE_ROOM = "side_room"
    SHOP = "shop"


class _Enemy(Struct, frozen=True, forbid_unknown_fields=True):
    id: int
    stomp: Stomp


class _Hazard(Struct, frozen=True, forbid_unknown_fields=True):
    id: int


class _Pickup(Struct, frozen=True, forbid_unknown_fields=True):
    id: int
    category: PickupCategory
    parent: str | None = None


class _Upgrades(Struct, frozen=True, forbid_unknown_fields=True):
    names: list[str]


class _GameData(Struct, frozen=True, forbid_unknown_fields=True):
    upgrades: _Upgrades
    enemies: dict[str, _Enemy]
    hazards: dict[str, _Hazard]
    terrain_hazards: dict[str, _Hazard]
    pickups: dict[str, _Pickup]
    extras: dict[str, _Hazard]


# a typo or an unknown category should fail here at import, not mid-training
_DATA: Final[_GameData] = msgspec.toml.decode((Path(__file__).parent / "downwell.toml").read_bytes(), type=_GameData)

STOMP_RULE: Final[dict[int, Stomp]] = {enemy.id: enemy.stomp for enemy in _DATA.enemies.values()}
HAZARD_IDS: Final[frozenset[int]] = frozenset(hazard.id for hazard in _DATA.hazards.values())
TERRAIN_HAZARD_IDS: Final[frozenset[int]] = frozenset(hazard.id for hazard in _DATA.terrain_hazards.values())
PICKUP_CATEGORY: Final[dict[int, PickupCategory]] = {pickup.id: pickup.category for pickup in _DATA.pickups.values()}
UPGRADE_NAMES: Final[list[str]] = _DATA.upgrades.names
GEM_ID: Final[int] = _DATA.extras["objCurrency"].id
DRONE_ID: Final[int] = _DATA.extras["objDrone"].id
