from dataclasses import dataclass
import time
from typing import Final

from loguru import logger

from downwell_ai.consts import GAME_FPS
from downwell_ai.consts import PROCESS_NAME
from downwell_ai.utils.exceptions import GameTimeoutError

from ._game_process import GameProcess
from ._game_process import open_game_process
from ._state import MemoryState
from ._state import parse_snapshot


_ROOM_MAIN: Final[int] = 11

_LOCKSTEP_SPEED: Final[int] = 10000

_CONTROL_TIMEOUT_MS: Final[int] = 2000

_STEP_TIMEOUT_SECONDS: Final[float] = 2.0
_RESET_TIMEOUT_SECONDS: Final[float] = 10.0
_RELEASE_TIMEOUT_SECONDS: Final[float] = 5.0


@dataclass(frozen=True)
class Lockstep:
    """
    Drives the patched game frame by frame: input goes in through `dqn_act`, state comes out of the packed array.

    Do not use standalone; use `attach()`.
    """

    _game: GameProcess

    def _wait_for(self, field: str, value: int, waiting_for: str, timeout: float) -> None:
        addr = self._game.address(field)
        deadline = time.perf_counter() + timeout

        while self._game.read_rvalue(addr) != value:
            if time.perf_counter() > deadline:
                raise GameTimeoutError(waiting_for, timeout)

    def read(self) -> MemoryState:
        """Sample the latest exported state. Raises `FieldResolveError` or `StateLayoutError` if the patch is missing or stale."""
        return parse_snapshot(self._game.read_snapshot())

    def take_control(self, *, is_headless: bool = True, fps: int | None = None) -> None:
        """
        Switch the game to lockstep: it simulates only when `step()` releases it, at whatever pace the agent sets.

        `fps` caps the frame rate inside a step, e.g. `GAME_FPS` to watch at real speed; `None` lifts the cap.
        """
        self._game.write_field("timeout_ms", _CONTROL_TIMEOUT_MS)
        self._game.write_field("headless", 1 if is_headless else 0)
        self._game.write_field("speed", fps or _LOCKSTEP_SPEED)
        self._game.write_field("ctrl", 1)

        logger.debug(f"took control (headless={is_headless})")

    def release_control(self) -> None:
        """Give input back to the keyboard and restore drawing, VSync and the native frame rate."""
        self._game.write_field("ctrl", 0)
        self._game.write_field("headless", 0)

        # speed 0 alone keeps the last forced room_speed, so force the native rate for a full frame first
        self._game.write_field("speed", GAME_FPS)
        frame_addr = self._game.address("frame")
        start = self._game.read_rvalue(frame_addr)
        deadline = time.perf_counter() + _RELEASE_TIMEOUT_SECONDS

        while self._game.read_rvalue(frame_addr) < start + 2:
            if time.perf_counter() > deadline:
                logger.warning(f"game did not advance within {_RELEASE_TIMEOUT_SECONDS:.0f}s of release, speed left forced at {GAME_FPS}")
                return

        self._game.write_field("speed", 0)
        logger.debug("released control")

    def step(self, inputs: int, repeat: int = 1) -> MemoryState:
        """
        Hold `inputs` (`INPUT_LEFT | INPUT_RIGHT | INPUT_JUMP` bits) for `repeat` frames and return the state after them.

        Needs `take_control()` first. The game stays blocked until the next call, so the returned state is never torn.
        Raises `GameTimeoutError` if the game does not finish the step within `_STEP_TIMEOUT_SECONDS`.
        """
        step_id = int(self._game.read_field("done")) + 1

        self._game.write_field("act", inputs)
        self._game.write_field("repeat", repeat)
        self._game.write_field("step", step_id)  # last, this is what releases the game

        self._wait_for("ready", step_id, "a lockstep step", _STEP_TIMEOUT_SECONDS)
        return self.read()

    def reset(self) -> MemoryState:
        """
        Start a fresh run and step with no input until the player is alive in rmMain 1-1.

        Raises `GameTimeoutError` if the player does not spawn within `_RESET_TIMEOUT_SECONDS`.
        """
        # the room change skips the rest of the reset frame
        # the game reports the step ready on the next one
        self._game.write_field("reset", 1)

        deadline = time.perf_counter() + _RESET_TIMEOUT_SECONDS
        state = self.step(0)

        # the first frames can still carry the previous run's level, wait for 1-1 too
        while not (state.room == _ROOM_MAIN and state.has_player and not state.death and state.area == 1 and state.level == 1):
            if time.perf_counter() > deadline:
                waiting_for = "the player to spawn after reset"
                raise GameTimeoutError(waiting_for, _RESET_TIMEOUT_SECONDS)

            state = self.step(0)

        return state

    def close(self) -> None:
        """Detach from the game. This object must not be used after calling this."""
        self._game.close()


def attach(proc_name: str = PROCESS_NAME) -> Lockstep | None:
    """Attach to the patched game."""
    game = open_game_process(proc_name)
    return Lockstep(game) if game is not None else None
