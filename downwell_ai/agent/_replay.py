from collections import deque
import math
from typing import Final

from msgspec import Struct
import numpy as np
import torch

from downwell_ai.env.observation import GRID_SHAPE
from downwell_ai.env.observation import VEC_DIM


_GRID_CELLS: Final[int] = math.prod(GRID_SHAPE)
_PACKED_GRID_BYTES: Final[int] = -(-_GRID_CELLS // 8)


class Batch(Struct, frozen=True):
    """Sampled transitions as tensors on the buffer's device, one row per transition."""

    grids: torch.Tensor
    vecs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_grids: torch.Tensor
    next_vecs: torch.Tensor
    discounts: torch.Tensor


class ReplayBuffer:
    """Fixed-size ring buffer of transitions, overwriting the oldest once full."""

    def __init__(self, capacity: int, seed: int, device: torch.device | None = None) -> None:
        self.__grids = np.zeros((capacity, _PACKED_GRID_BYTES), dtype=np.uint8)
        self.__next_grids = np.zeros((capacity, _PACKED_GRID_BYTES), dtype=np.uint8)
        self.__vecs = np.zeros((capacity, VEC_DIM), dtype=np.float32)
        self.__next_vecs = np.zeros((capacity, VEC_DIM), dtype=np.float32)

        self.__actions = np.zeros(capacity, dtype=np.int64)
        self.__rewards = np.zeros(capacity, dtype=np.float32)
        self.__discounts = np.zeros(capacity, dtype=np.float32)

        self.__capacity = capacity
        self.__idx = 0
        self.__size = 0

        self.__device = device or torch.device("cpu")
        self.__rng = np.random.default_rng(seed)

    def push(
        self,
        obs: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: dict[str, np.ndarray],
        discount: float,
    ) -> None:
        """Store one transition."""
        idx = self.__idx

        self.__grids[idx] = np.packbits(obs["grid"])
        self.__vecs[idx] = obs["vec"]
        self.__next_grids[idx] = np.packbits(next_obs["grid"])
        self.__next_vecs[idx] = next_obs["vec"]

        self.__actions[idx] = action
        self.__rewards[idx] = reward
        self.__discounts[idx] = discount

        self.__idx = (idx + 1) % self.__capacity
        self.__size = min(self.__size + 1, self.__capacity)

    def sample(self, batch_size: int) -> Batch:
        """Draw `batch_size` transitions uniformly, with replacement."""
        indices = self.__rng.integers(0, self.__size, size=batch_size)

        def to_device(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(self.__device, non_blocking=True)

        def unpack(packed: np.ndarray) -> np.ndarray:
            return np.unpackbits(packed[indices], axis=1, count=_GRID_CELLS).reshape(batch_size, *GRID_SHAPE)

        return Batch(
            grids=to_device(unpack(self.__grids)),
            vecs=to_device(self.__vecs[indices]),
            actions=to_device(self.__actions[indices]),
            rewards=to_device(self.__rewards[indices]),
            next_grids=to_device(unpack(self.__next_grids)),
            next_vecs=to_device(self.__next_vecs[indices]),
            discounts=to_device(self.__discounts[indices]),
        )

    def __len__(self) -> int:
        return self.__size


class NStepCollector:
    """
    Turn single steps into `n`-step transitions before they reach the buffer.

    Each stored reward sums the next n rewards discounted by `gamma`, and its discount is `gamma ** n` for
    bootstrapping from the state `n` steps later. When an episode ends the remaining steps are flushed
    with shorter spans.
    """

    def __init__(self, buffer: ReplayBuffer, n_steps: int, gamma: float) -> None:
        self.__buffer = buffer
        self.__n_steps = n_steps
        self.__gamma = gamma
        self.__pending: deque[tuple[dict[str, np.ndarray], int, float]] = deque()

    def _push_oldest(self, next_obs: dict[str, np.ndarray], *, is_terminated: bool) -> None:
        span = len(self.__pending)
        n_step_reward = sum(self.__gamma**i * reward for i, (_, _, reward) in enumerate(self.__pending))
        obs, action, _ = self.__pending.popleft()
        discount = 0.0 if is_terminated else self.__gamma**span
        self.__buffer.push(obs, action, n_step_reward, next_obs, discount)

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: dict[str, np.ndarray],
        *,
        is_terminated: bool,
        is_truncated: bool,
    ) -> None:
        """Record one step, pushing every transition whose n-step span is now complete."""
        self.__pending.append((obs, action, reward))

        if is_terminated or is_truncated:
            while self.__pending:
                self._push_oldest(next_obs, is_terminated=is_terminated)
            return

        if len(self.__pending) == self.__n_steps:
            self._push_oldest(next_obs, is_terminated=False)
