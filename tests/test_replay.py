import pytest
import torch

from conftest import rand_grid
from conftest import rand_vec
from downwell_ai.agent import NStepCollector
from downwell_ai.agent import ReplayBuffer


_CAPACITY = 20


def _obs() -> dict:
    return {"grid": rand_grid(), "vec": rand_vec()}


def _push(buf: ReplayBuffer, action: int = 0) -> None:
    buf.push(_obs(), action, 0.0, _obs(), discount=0.99)


def test_partially_filled_buf_never_samples_empty_slots():
    buf = ReplayBuffer(_CAPACITY, seed=0)
    _push(buf, action=3)

    batch = buf.sample(64)

    assert torch.all(batch.actions == 3)


def test_full_buf_overwrites_the_oldest_transition():
    buf = ReplayBuffer(_CAPACITY, seed=0)
    _push(buf, action=1)

    for _ in range(_CAPACITY):
        _push(buf, action=2)

    batch = buf.sample(256)

    assert torch.all(batch.actions == 2)


def test_next_obs_belongs_to_the_same_transition():
    buf = ReplayBuffer(_CAPACITY, seed=0)
    first, second, unrelated = _obs(), _obs(), _obs()
    buf.push(first, 1, 0.5, second, discount=0.0)
    # a following transition from another episode must not leak into the first one's next observation
    buf.push(unrelated, 2, 0.0, unrelated, discount=0.99)

    batch = buf.sample(64)
    rows = batch.actions == 1

    assert torch.equal(batch.next_vecs[rows][0], torch.from_numpy(second["vec"]))
    assert torch.all(batch.discounts[rows] == 0.0)


def test_grids_come_back_exactly_as_pushed():
    buf = ReplayBuffer(_CAPACITY, seed=0)
    obs, next_obs = _obs(), _obs()
    buf.push(obs, 0, 0.0, next_obs, discount=0.99)

    batch = buf.sample(1)

    assert torch.equal(batch.grids[0], torch.from_numpy(obs["grid"]))
    assert torch.equal(batch.next_grids[0], torch.from_numpy(next_obs["grid"]))


def _collect(rewards: list[float], *, is_terminated: bool, is_truncated: bool = False) -> dict[int, tuple[float, float]]:
    """Feed one episode with action i at step i through a 3-step collector, returning each action's (reward, discount)."""
    buf = ReplayBuffer(_CAPACITY, seed=0)
    collector = NStepCollector(buf, n_steps=3, gamma=0.5)

    for action, reward in enumerate(rewards):
        is_last = action == len(rewards) - 1
        collector.add(_obs(), action, reward, _obs(), is_terminated=is_last and is_terminated, is_truncated=is_last and is_truncated)

    batch = buf.sample(256)
    return {int(a): (float(r), float(d)) for a, r, d in zip(batch.actions, batch.rewards, batch.discounts, strict=True)}


def test_n_step_reward_sums_the_next_n_rewards_discounted():
    transitions = _collect([1.0, 2.0, 4.0, 8.0, 0.0], is_terminated=False, is_truncated=True)

    assert transitions[0] == pytest.approx((1.0 + 0.5 * 2.0 + 0.25 * 4.0, 0.125))
    assert transitions[1] == pytest.approx((2.0 + 0.5 * 4.0 + 0.25 * 8.0, 0.125))


def test_episode_end_flushes_shorter_spans_and_death_never_bootstraps():
    transitions = _collect([1.0, 2.0, 4.0, 8.0], is_terminated=True)

    assert transitions[2] == pytest.approx((4.0 + 0.5 * 8.0, 0.0))
    assert transitions[3] == pytest.approx((8.0, 0.0))


def test_truncation_flushes_shorter_spans_that_still_bootstrap():
    transitions = _collect([1.0, 2.0, 4.0, 8.0], is_terminated=False, is_truncated=True)

    assert transitions[2] == pytest.approx((4.0 + 0.5 * 8.0, 0.25))
    assert transitions[3] == pytest.approx((8.0, 0.5))
