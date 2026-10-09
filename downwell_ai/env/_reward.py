from typing import Final

from downwell_ai.config import RewardConfig
from downwell_ai.consts import GAME_FPS
from downwell_ai.env.memory import MemoryState


_LEVELS_PER_AREA: Final[int] = 3


def _overall_level(state: MemoryState) -> int:
    return (state.area - 1) * _LEVELS_PER_AREA + state.level


def is_level_complete(previous: MemoryState, state: MemoryState) -> bool:
    """Whether the player left `previous`'s level for the next one, including from one area into the next."""
    return _overall_level(state) > _overall_level(previous)


def is_dead(state: MemoryState) -> bool:
    """Whether the run ended with the player dead."""
    return state.death != 0 or state.hp <= 0


class ProgressTracker:
    """Best depth reached on the current level and the game time since it last improved."""

    def _anchor(self, state: MemoryState) -> None:
        self.__level = _overall_level(state)
        self.__room = state.room
        self.__best_y = state.ypos
        self.__frames_since_best = 0

    def __init__(self, initial: MemoryState) -> None:
        self.__depth = 0.0
        self._anchor(initial)

    @property
    def depth(self) -> float:
        """Total px of new best depth over every level of the episode."""
        return self.__depth

    @property
    def seconds_since_best(self) -> float:
        """Game seconds since best depth last improved on this level."""
        return self.__frames_since_best / GAME_FPS

    def update(self, state: MemoryState, frames: int) -> float:
        """Advance by `frames` game frames ending in `state` and return the px of new best depth."""
        if _overall_level(state) != self.__level:
            # new level, ypos restarts near the top
            self._anchor(state)
            return 0.0

        # side rooms, shops and the upgrade menu are worth visiting, time there must not count as stalling
        is_detour = state.in_side_room or state.shop or state.menu

        # gamemaker y grows downwards
        if state.room != self.__room or state.ypos <= self.__best_y:
            if not is_detour:
                self.__frames_since_best += frames
            return 0.0

        progress_px = state.ypos - self.__best_y

        self.__best_y = state.ypos
        self.__depth += progress_px
        self.__frames_since_best = 0

        return progress_px


def compute_reward(previous: MemoryState, state: MemoryState, progress_px: float, config: RewardConfig) -> tuple[float, dict[str, float]]:
    """
    Score the step from `previous` to `state`, `progress_px` being the new best depth from `ProgressTracker.update`.

    Returns the scaled total and its per-term breakdown, which sums to the total.
    - Gems only count increases, so spending them in a shop costs nothing.
    - Healing pays per hp gained, so hp bought in a shop or from an upgrade pays now rather than as hits avoided levels later.
    - Max ammo only counts increases, like gems.
    - Kills are the combo's increases: the game adds one per kill, and kills happen in the air where the combo runs.
    - A combo that ends (lands) at N pays N squared, so chains are worth more than the same kills apart.
    """
    is_fatal = is_dead(state)
    is_hit = not is_fatal and state.hp < previous.hp
    healed = max(state.hp - previous.hp, 0) if not is_fatal else 0
    ammo_gained = max(state.ammo_max - previous.ammo_max, 0)
    kills = max(state.combo - previous.combo, 0)
    ended_combo = previous.combo if previous.combo > 0 and state.combo == 0 and not is_fatal else 0

    components = {
        "progress": progress_px * config.depth_reward,
        "gem": max(state.gems - previous.gems, 0) * config.gem_reward,
        "hit": config.hit_penalty if is_hit else 0.0,
        "heal": healed * config.heal_reward,
        "ammo": ammo_gained * config.ammo_reward,
        "death": config.death_penalty if is_fatal else 0.0,
        "level": config.level_reward if is_level_complete(previous, state) else 0.0,
        "kill": kills * config.kill_reward,
        "combo": ended_combo**2 * config.combo_reward,
    }
    components = {name: value * config.scale for name, value in components.items()}

    return sum(components.values()), components
