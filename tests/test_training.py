import numpy as np

from downwell_ai._training import play_evaluation_episode


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


class _RecordingAgent:
    def __init__(self) -> None:
        self.epsilons: list[float] = []

    def select_action(self, _obs: dict, eps: float) -> int:
        self.epsilons.append(eps)
        return 0


class _ThreeStepEnv:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self) -> tuple[dict, dict]:
        self.steps = 0
        return {}, {}

    def step(self, _action: int) -> tuple[dict, float, bool, bool, dict]:
        self.steps += 1
        info = {"reward_components": {"progress": 1.0}, "area": 1, "level": 1, "depth": float(self.steps)}
        return {}, 1.0, self.steps == 3, False, info

    def render(self) -> np.ndarray:
        return _frame(self.steps)


def test_evaluation_episode_frames_include_the_state_it_ended_on():
    _, frames = play_evaluation_episode(_RecordingAgent(), _ThreeStepEnv(), step=10, eps=0.0)  # ty: ignore[invalid-argument-type]

    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1, 2, 3]
