from typing import Annotated

from msgspec import Meta
from msgspec import Struct


_Positive = Meta(gt=0)
_Unit = Meta(ge=0, le=1)


class RewardConfig(Struct, frozen=True, forbid_unknown_fields=True):
    """Reward coefficients, all applied before the single global `scale`. Every field must be set in the training toml."""

    depth_reward: float
    gem_reward: float
    hit_penalty: float
    heal_reward: float
    ammo_reward: float
    death_penalty: float
    level_reward: float
    kill_reward: float
    combo_reward: float
    scale: Annotated[float, _Positive]


class TrainingConfig(Struct, frozen=True, forbid_unknown_fields=True):
    """Hyperparameters for one training run, decoded from a toml such as `configs/default.toml` with no fallback defaults."""

    seed: int
    total_steps: Annotated[int, _Positive]
    learning_rate: Annotated[float, _Positive]
    gamma: Annotated[float, _Unit]
    n_steps: Annotated[int, _Positive]
    buf_cap: Annotated[int, _Positive]
    batch_size: Annotated[int, _Positive]
    learning_starts: Annotated[int, Meta(ge=0)]
    train_frequency: Annotated[int, _Positive]
    target_update_interval: Annotated[int, _Positive]
    max_grad_norm: Annotated[float, _Positive]
    eps_start: Annotated[float, _Unit]
    eps_end: Annotated[float, _Unit]
    resume_eps_start: Annotated[float, _Unit]
    exploration_steps: Annotated[int, _Positive]
    eval_interval_episodes: Annotated[int, _Positive]
    eval_eps: Annotated[float, _Unit]
    frame_skip: Annotated[int, _Positive]
    headless: bool
    max_episode_steps: Annotated[int, _Positive]
    stall_timeout_seconds: Annotated[float, Meta(ge=0)]
    reward: RewardConfig
