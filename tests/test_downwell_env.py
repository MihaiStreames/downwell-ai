from typing import Any

import pytest

from conftest import make_config
from conftest import make_reward_config
from conftest import make_state
from downwell_ai.consts import ACTION_NONE
from downwell_ai.consts import ACTION_RIGHT
from downwell_ai.consts import INPUT_RIGHT
from downwell_ai.env import downwell_env
from downwell_ai.env._game_data import UPGRADE_NAMES
from downwell_ai.env.downwell_env import DownwellEnv
from downwell_ai.env.memory import MemoryState
from downwell_ai.env.observation import OBS_SPACE
from downwell_ai.env.observation import VEC_MENU
from downwell_ai.utils.exceptions import GameTimeoutError


class _FakeMemory:
    def __init__(self, states: list[MemoryState]) -> None:
        self.states = states
        self.steps: list[tuple[int, int]] = []

    def _next(self) -> MemoryState:
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def take_control(self, *, is_headless: bool = True, fps: int | None = None) -> None:
        pass

    def release_control(self) -> None:
        pass

    def reset(self) -> MemoryState:
        return self._next()

    def step(self, inputs: int, repeat: int = 1) -> MemoryState:
        self.steps.append((inputs, repeat))
        return self._next()

    def close(self) -> None:
        pass


def _make_env(monkeypatch: pytest.MonkeyPatch, states: list[MemoryState], **config: Any) -> DownwellEnv:  # noqa: ANN401 (any TrainingConfig field)
    monkeypatch.setattr(downwell_env, "attach", lambda: _FakeMemory(states))
    # step counts in these tests assume 4 frames per step
    return DownwellEnv(make_config(**{"frame_skip": 4, **config}))


def test_reset_obs_matches_declared_space(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, [make_state()])
    obs, _ = env.reset()

    assert OBS_SPACE.contains(obs)


def test_step_limit_truncates_not_terminates(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, [make_state()], max_episode_steps=3)
    env.reset()

    flags = [env.step(ACTION_NONE)[2:4] for _ in range(3)]

    assert flags == [(False, False), (False, False), (False, True)]


def test_death_on_the_last_allowed_step_is_terminated_not_truncated(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, [make_state(), make_state(), make_state(hp=0, death=1)], max_episode_steps=2)
    env.reset()

    env.step(ACTION_NONE)
    _, _, terminated, truncated, _ = env.step(ACTION_NONE)

    assert terminated
    assert not truncated


def test_level_clear_pays_bonus_and_keeps_the_episode_going(monkeypatch: pytest.MonkeyPatch):
    states = [make_state(), make_state(has_player=0), make_state(has_player=0), make_state(level=2)]
    env = _make_env(monkeypatch, states, reward=make_reward_config(level_reward=10.0))
    env.reset()

    _, _, terminated, truncated, info = env.step(ACTION_NONE)

    assert not terminated
    assert not truncated
    assert info["level"] == 2
    assert info["reward_components"]["level"] == 10.0


def test_transition_that_never_ends_times_out(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, [make_state(), make_state(has_player=0)])
    env.reset()

    with pytest.raises(GameTimeoutError, match="next level"):
        env.step(ACTION_NONE)


def test_stalling_truncates_once_the_stall_timeout_passes(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, [make_state()], stall_timeout_seconds=1.0)  # 60 game frames, 15 steps
    env.reset()

    truncated = [env.step(ACTION_NONE)[3] for _ in range(15)]

    assert truncated == [False] * 14 + [True]


def test_zero_stall_timeout_never_truncates(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, [make_state()], stall_timeout_seconds=0.0)
    env.reset()

    assert not any(env.step(ACTION_NONE)[3] for _ in range(100))


def test_new_depth_restarts_the_stall_clock(monkeypatch: pytest.MonkeyPatch):
    states = [make_state(ypos=float(y)) for y in [0] + [0] * 10 + [50] + [50] * 10]
    env = _make_env(monkeypatch, states, stall_timeout_seconds=1.0)
    env.reset()

    assert not any(env.step(ACTION_NONE)[3] for _ in range(21))


def test_dying_during_a_level_transition_ends_the_episode_instead_of_waiting_out_the_timeout(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, [make_state(), make_state(has_player=0), make_state(has_player=0, hp=0, death=1)])
    env.reset()

    _, _, terminated, truncated, _ = env.step(ACTION_NONE)

    assert terminated
    assert not truncated


def test_upgrade_menu_is_handed_to_the_agent_and_buttons_are_pressed_once(monkeypatch: pytest.MonkeyPatch):
    menu = make_state(has_player=0, room=16, menu=1, offers=[13, 6, 9, -1])
    memory = _FakeMemory([make_state(), make_state(has_player=0, room=16), menu, menu, make_state(level=2)])
    monkeypatch.setattr(downwell_env, "attach", lambda: memory)
    env = DownwellEnv(make_config(frame_skip=4))
    env.reset()

    obs, _, terminated, _, _ = env.step(ACTION_NONE)
    memory.steps.clear()
    env.step(ACTION_RIGHT)

    assert not terminated
    assert obs["vec"][VEC_MENU][0] == 1.0
    assert memory.steps == [(INPUT_RIGHT, 1), (0, 3)]


def test_an_upgrade_pick_reports_what_the_menu_offered(monkeypatch: pytest.MonkeyPatch):
    owned = [0] * len(UPGRADE_NAMES)
    picked_apple = [1 if upgrade == 13 else 0 for upgrade in range(len(UPGRADE_NAMES))]
    menu = make_state(has_player=0, room=16, menu=1, offers=[13, 6, 9, -1], owned=owned)
    memory = _FakeMemory([make_state(owned=owned), menu, menu, make_state(level=2, owned=picked_apple)])
    monkeypatch.setattr(downwell_env, "attach", lambda: memory)
    env = DownwellEnv(make_config(frame_skip=4))
    env.reset()

    *_, on_menu = env.step(ACTION_NONE)
    *_, after_pick = env.step(ACTION_NONE)

    assert on_menu["upgrade"] is None
    assert after_pick["upgrade"] == {"area": 1, "level": 1, "offered": ["apple", "youth", "drone"], "picked": "apple"}


def test_entering_a_side_room_is_reported_once(monkeypatch: pytest.MonkeyPatch):
    states = [make_state(), make_state(in_side_room=1, room=9), make_state(in_side_room=1, room=9)]
    env = _make_env(monkeypatch, states)
    env.reset()

    infos = [env.step(ACTION_NONE)[4] for _ in range(2)]

    assert [info["entered_side_room"] for info in infos] == [True, False]
    assert all(info["in_side_room"] for info in infos)
