import itertools

import pytest

from conftest import make_reward_config
from conftest import make_state
from downwell_ai.env._reward import ProgressTracker
from downwell_ai.env._reward import compute_reward


_CONFIG = make_reward_config(depth_reward=0.01)


def _progress_rewards(ys: list[float]) -> list[float]:
    states = [make_state(ypos=y) for y in ys]
    tracker = ProgressTracker(states[0])
    rewards = []

    for previous, state in itertools.pairwise(states):
        _, components = compute_reward(previous, state, tracker.update(state, frames=4), _CONFIG)
        rewards.append(components["progress"])

    return rewards


def test_only_depth_beyond_previous_best_pays():
    rewards = _progress_rewards([0, 100, 40, 100, 130])

    assert rewards == pytest.approx([1.0, 0.0, 0.0, 0.3])


def test_hit_penalized_once_regardless_of_damage():
    _, components = compute_reward(make_state(hp=4), make_state(hp=2), 0.0, _CONFIG)

    assert components["hit"] == _CONFIG.hit_penalty


def test_fatal_hit_pays_death_not_hit():
    _, components = compute_reward(make_state(hp=1), make_state(hp=0), 0.0, _CONFIG)

    assert components["hit"] == 0.0
    assert components["death"] == _CONFIG.death_penalty


def test_death_flag_counts_even_with_hp_left():
    _, components = compute_reward(make_state(hp=2), make_state(hp=2, death=1), 0.0, _CONFIG)

    assert components["death"] == _CONFIG.death_penalty


def test_healing_pays_per_hp_gained():
    config = make_reward_config(heal_reward=2.5)

    _, components = compute_reward(make_state(hp=2), make_state(hp=4), 0.0, config)

    assert components["heal"] == pytest.approx(5.0)


def test_a_hit_then_a_heal_of_the_same_hp_nets_nothing():
    config = make_reward_config(hit_penalty=-2.5, heal_reward=2.5)

    _, hit = compute_reward(make_state(hp=4), make_state(hp=3), 0.0, config)
    _, healed = compute_reward(make_state(hp=3), make_state(hp=4), 0.0, config)

    assert hit["hit"] + healed["heal"] == 0.0


def test_max_ammo_pays_on_increase_and_losing_it_costs_nothing():
    config = make_reward_config(ammo_reward=1.0)

    _, gained = compute_reward(make_state(ammo_max=8), make_state(ammo_max=10), 0.0, config)
    _, lost = compute_reward(make_state(ammo_max=10), make_state(ammo_max=8), 0.0, config)

    assert gained["ammo"] == pytest.approx(2.0)
    assert lost["ammo"] == 0.0


def test_side_room_coordinates_never_pay_progress():
    tracker = ProgressTracker(make_state(ypos=100.0, room=7))

    in_side_room = tracker.update(make_state(ypos=5000.0, room=9), frames=4)
    back_in_level = tracker.update(make_state(ypos=150.0, room=7), frames=4)

    assert in_side_room == 0.0
    assert back_in_level == pytest.approx(50.0)


def test_new_level_restarts_depth_from_its_spawn():
    tracker = ProgressTracker(make_state(ypos=0.0, level=1))
    tracker.update(make_state(ypos=3600.0, level=1), frames=4)

    on_arrival = tracker.update(make_state(ypos=20.0, level=2, room=8), frames=4)
    first_descent = tracker.update(make_state(ypos=70.0, level=2, room=8), frames=4)

    assert on_arrival == 0.0
    assert first_descent == pytest.approx(50.0)
    assert tracker.depth == pytest.approx(3650.0)


def test_scale_applies_to_every_term_and_total():
    scaled = make_reward_config(depth_reward=0.01, gem_reward=0.2, hit_penalty=-1.0, scale=0.1)

    reward, components = compute_reward(make_state(hp=4), make_state(hp=3, gems=5), 100.0, scaled)

    assert components["hit"] == pytest.approx(-0.1)
    assert reward == pytest.approx(sum(components.values()))
    assert reward == pytest.approx(0.1 * (1.0 + 1.0 - 1.0))


def test_clearing_the_last_level_of_an_area_pays_the_level_bonus():
    # the game's level counter restarts at 1 in each area, 1-3 -> 2-1 reads as level 3 -> 1
    _, components = compute_reward(make_state(area=1, level=3), make_state(area=2, level=1), 0.0, _CONFIG)

    assert components["level"] == _CONFIG.level_reward


def test_gems_pay_on_pickup_and_spending_costs_nothing():
    _, picked_up = compute_reward(make_state(gems=10), make_state(gems=15), 0.0, _CONFIG)
    _, spent = compute_reward(make_state(gems=50), make_state(gems=20), 0.0, _CONFIG)

    assert picked_up["gem"] == pytest.approx(5 * _CONFIG.gem_reward)
    assert spent["gem"] == 0.0


def test_time_off_the_level_outside_a_detour_counts_towards_the_stall_clock():
    tracker = ProgressTracker(make_state(room=7))
    tracker.update(make_state(room=7), frames=60)

    tracker.update(make_state(room=9), frames=60)

    assert tracker.seconds_since_best == pytest.approx(2.0)


@pytest.mark.parametrize("detour", [{"in_side_room": 1, "room": 9}, {"in_side_room": 1, "shop": 1, "room": 9}, {"menu": 1, "room": 16}])
def test_side_rooms_shops_and_menus_pause_the_stall_clock(detour: dict[str, int]):
    tracker = ProgressTracker(make_state(room=7))
    tracker.update(make_state(room=7), frames=60)

    tracker.update(make_state(**detour), frames=600)

    assert tracker.seconds_since_best == pytest.approx(1.0)


def test_passing_a_shop_without_going_in_counts_towards_the_stall_clock():
    tracker = ProgressTracker(make_state())
    tracker.update(make_state(), frames=60)

    tracker.update(make_state(shop=1), frames=60)

    assert tracker.seconds_since_best == pytest.approx(2.0)


def test_each_combo_increase_pays_a_kill():
    config = make_reward_config(kill_reward=0.5)

    _, components = compute_reward(make_state(combo=2), make_state(combo=4), 0.0, config)

    assert components["kill"] == pytest.approx(1.0)


def test_an_ended_combo_pays_its_length_squared():
    config = make_reward_config(combo_reward=0.05)

    _, components = compute_reward(make_state(combo=6), make_state(combo=0), 0.0, config)

    assert components["combo"] == pytest.approx(0.05 * 36)
    assert components["kill"] == 0.0


def test_a_combo_cut_short_by_death_pays_nothing():
    config = make_reward_config(combo_reward=0.05)

    _, components = compute_reward(make_state(combo=6, hp=1), make_state(combo=0, hp=0), 0.0, config)

    assert components["combo"] == 0.0
