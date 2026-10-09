import time
from typing import Any
from typing import Final

from gymnasium import Env
from gymnasium.spaces import Discrete
from loguru import logger
import numpy as np

from downwell_ai.config import TrainingConfig
from downwell_ai.consts import ACTION_INPUTS
from downwell_ai.consts import GAME_FPS
from downwell_ai.consts import INPUT_JUMP
from downwell_ai.consts import N_ACTIONS
from downwell_ai.env._game_data import UPGRADE_NAMES
from downwell_ai.env._render import render_obs
from downwell_ai.env._reward import ProgressTracker
from downwell_ai.env._reward import compute_reward
from downwell_ai.env._reward import is_dead
from downwell_ai.env._reward import is_level_complete
from downwell_ai.env.memory import Lockstep
from downwell_ai.env.memory import MemoryState
from downwell_ai.env.memory import attach
from downwell_ai.env.observation import OBS_SPACE
from downwell_ai.env.observation import encode_obs
from downwell_ai.utils.exceptions import GameTimeoutError


type Observation = dict[str, np.ndarray]

_ATTACH_TIMEOUT_SECONDS: Final[float] = 30.0
_TRANSITION_TIMEOUT_FRAMES: Final[int] = 30 * GAME_FPS  # level transitions, in game frames
_MENU_TAP_FRAMES: Final[int] = GAME_FPS
_ROOM_RESULT: Final[int] = 16  # rmResult, the level-end screen with the upgrade menu


def _is_playable(state: MemoryState) -> bool:
    return state.has_player == 1 and not is_dead(state)


class DownwellEnv(Env[Observation, np.int64]):
    """
    Gymnasium environment around a live, patched Downwell process, one episode per run.

    NOTE: The game runs in lockstep, meaning it simulates `frame_skip` frames with the chosen buttons held, then blocks until the next step.
    Raises `FieldResolveError` when memory becomes unreadable and `GameTimeoutError` when the game stops advancing.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, config: TrainingConfig, render_mode: str | None = None, *, is_realtime: bool = False) -> None:
        self.obs_space = OBS_SPACE
        self.action_space = Discrete(N_ACTIONS)
        self.render_mode = render_mode

        self.__config = config
        self.__is_realtime = is_realtime

        self.__game: Lockstep | None = None

        self.__progress: ProgressTracker | None = None

        self.__prev: MemoryState | None = None
        self.__obs: Observation | None = None

        self.__step_count = 0
        self.__level_start_step = 0
        self.__offers: list[int] = []

    def _attach(self) -> Lockstep:
        if self.__game is not None:
            return self.__game

        deadline = time.monotonic() + _ATTACH_TIMEOUT_SECONDS

        while (game := attach()) is None:
            if time.monotonic() > deadline:
                waiting_for = "the downwell process"
                raise GameTimeoutError(waiting_for, _ATTACH_TIMEOUT_SECONDS)

            time.sleep(1.0)

        game.take_control(is_headless=self.__config.headless, fps=GAME_FPS if self.__is_realtime else None)
        self.__game = game

        return game

    def _play_through_transition(self, game: Lockstep, state: MemoryState) -> MemoryState:
        # wait until play resumes or the upgrade menu wants a pick, which goes back to the agent
        frames = 0
        while not _is_playable(state) and not is_dead(state) and not state.menu:
            if frames >= _TRANSITION_TIMEOUT_FRAMES:
                waiting_for = "the next level to load"
                raise GameTimeoutError(waiting_for, _TRANSITION_TIMEOUT_FRAMES / GAME_FPS)

            # the result screen opens its menu on its own and a jump there picks, anywhere else a tap skips a prompt
            inputs = 0 if state.room == _ROOM_RESULT else INPUT_JUMP
            state = game.step(inputs)
            state = game.step(0, repeat=_MENU_TAP_FRAMES - 1)
            frames += _MENU_TAP_FRAMES

        return state

    def _act(self, game: Lockstep, previous: MemoryState, inputs: int) -> MemoryState:
        if not previous.menu:
            return game.step(inputs, repeat=self.__config.frame_skip)

        game.step(inputs)

        return game.step(0, repeat=self.__config.frame_skip - 1) if self.__config.frame_skip > 1 else game.read()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Observation, dict[str, Any]]:
        """Start a fresh run at level 1-1."""
        # this only seeds np_random
        super().reset(seed=seed, options=options)
        game = self._attach()

        state = game.reset()
        logger.debug("episode started on level {}-{}", state.area, state.level)

        self.__progress = ProgressTracker(state)
        self.__prev = state
        self.__step_count = 0
        self.__level_start_step = 0
        self.__offers = []
        self.__obs = encode_obs(state)

        return self.__obs, {"area": state.area, "level": state.level}

    def step(self, action: np.int64 | int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Hold `action`'s buttons for `frame_skip` frames and score the result."""
        game, progress, previous = self.__game, self.__progress, self.__prev
        if game is None or progress is None or previous is None:
            msg = "step() called before reset()"
            raise RuntimeError(msg)

        state = self._act(game, previous, ACTION_INPUTS[int(action)])
        if not state.has_player and not is_dead(state) and not state.menu:
            state = self._play_through_transition(game, state)

        progress_px = progress.update(state, self.__config.frame_skip)
        reward, components = compute_reward(previous, state, progress_px, self.__config.reward)

        self.__prev = state
        self.__step_count += 1

        terminated = is_dead(state)
        is_stalled = 0 < self.__config.stall_timeout_seconds <= progress.seconds_since_best
        truncated = not terminated and (self.__step_count >= self.__config.max_episode_steps or is_stalled)

        if state.menu:
            self.__offers = state.offers

        is_hit = state.hp < previous.hp
        info: dict[str, Any] = {
            "reward_components": components,
            "area": state.area,
            "level": state.level,
            "depth": progress.depth,
            "is_hit": is_hit,
            "in_side_room": bool(state.in_side_room),
            "entered_side_room": bool(state.in_side_room and not previous.in_side_room),
            "entered_shop": bool(state.shop and not previous.shop),
            # outside a shop gems only ever grow
            "gems_spent": max(previous.gems - state.gems, 0) if state.shop or previous.shop else 0,
            "upgrade": None,
        }

        if is_hit:
            logger.debug("hit on level {}-{}, hp {} -> {}", previous.area, previous.level, previous.hp, state.hp)

        if is_level_complete(previous, state):
            level_steps = self.__step_count - self.__level_start_step
            logger.debug("level {}-{} cleared in {} steps, {} into the episode", previous.area, previous.level, level_steps, self.__step_count)
            self.__level_start_step = self.__step_count

        for upgrade, (before, after) in enumerate(zip(previous.owned, state.owned, strict=False)):
            if after > before:
                logger.debug("gained upgrade {} on level {}-{}", UPGRADE_NAMES[upgrade], previous.area, previous.level)
                offered = [UPGRADE_NAMES[offer] for offer in self.__offers if 0 <= offer < len(UPGRADE_NAMES)]
                info["upgrade"] = {"area": previous.area, "level": previous.level, "offered": offered, "picked": UPGRADE_NAMES[upgrade]}
                self.__offers = []

        self.__obs = encode_obs(state)

        return self.__obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Draw the latest observation, see `render_obs`."""
        if self.render_mode != "rgb_array" or self.__obs is None:
            return None

        return render_obs(self.__obs)

    def close(self) -> None:
        """Hand the game back to the keyboard with drawing and its own speed restored, then detach."""
        if self.__game is None:
            return

        self.__game.release_control()
        self.__game.close()
        self.__game = None
