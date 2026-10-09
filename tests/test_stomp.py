import numpy as np

from conftest import make_row
from conftest import make_rows
from conftest import make_state
from downwell_ai.env._stomp import stomp_flags
from downwell_ai.env.memory import FLAG_ENEMY
from downwell_ai.env.memory import FLAG_SOLID


SNAKE, SKELETON, GAS_WANDER, WANDER_SKULL, SHOPKEEPER = 111, 120, 123, 124, 284  # yes, falling, when_white, when_calm, no


def _flags(*rows: np.ndarray, **player: float) -> tuple[list[bool], list[bool]]:
    state = make_state(ypos=20.0, **player, instances=make_rows(*rows))
    now, from_above = stomp_flags(state)
    return now.tolist(), from_above.tolist()


def _enemy(obj: int, top: float = 100.0, **fields: float) -> np.ndarray:
    return make_row(obj, FLAG_ENEMY, (0, top, 7, top + 7), **fields)


def test_a_typical_enemy_below_the_player_is_stompable_now_and_from_above():
    assert _flags(_enemy(SNAKE)) == ([True], [True])


def test_a_typical_enemy_above_the_player_is_not_stompable_now():
    now, from_above = _flags(_enemy(SNAKE, top=-50.0))

    assert now == [False]
    assert from_above == [True]


def test_a_falling_only_enemy_needs_the_player_airborne_and_falling():
    assert _flags(_enemy(SKELETON), grounded=1, vy=0.0)[0] == [False]
    assert _flags(_enemy(SKELETON), grounded=0, vy=3.0)[0] == [True]


def test_gas_wander_is_only_stompable_while_white():
    assert _flags(_enemy(GAS_WANDER, white=0)) == ([False], [False])
    assert _flags(_enemy(GAS_WANDER, white=1)) == ([True], [True])


def test_an_angry_wander_skull_is_not_safe_to_stomp():
    assert _flags(_enemy(WANDER_SKULL, angry=1)) == ([False], [False])
    assert _flags(_enemy(WANDER_SKULL, angry=0)) == ([True], [True])


def test_enemies_that_never_stomp_and_non_enemies_are_never_stompable():
    wall = make_row(57, FLAG_SOLID, (0, 100, 7, 107))

    assert _flags(_enemy(SHOPKEEPER), wall) == ([False, False], [False, False])
