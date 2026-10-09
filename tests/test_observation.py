import numpy as np

from conftest import make_row
from conftest import make_rows
from conftest import make_state
from downwell_ai.env._render import render_obs
from downwell_ai.env.memory import FLAG_ENEMY
from downwell_ai.env.memory import FLAG_LISTED
from downwell_ai.env.memory import FLAG_SOLID
from downwell_ai.env.memory import N_UPGRADES
from downwell_ai.env.memory import OFFER_SLOTS
from downwell_ai.env.memory import SHOP_SLOTS
from downwell_ai.env.memory import MemoryState
from downwell_ai.env.observation import FRAME_SHAPE
from downwell_ai.env.observation import GRID_CELL_PX
from downwell_ai.env.observation import GRID_SHAPE
from downwell_ai.env.observation import MOTION_CHANNELS
from downwell_ai.env.observation import VEC_MENU
from downwell_ai.env.observation import VEC_SHOP
from downwell_ai.env.observation import Channel
from downwell_ai.env.observation import ObservationEncoder
from downwell_ai.env.observation import encode_grid
from downwell_ai.env.observation import encode_vector


SPIKE, BULLET, GEM = 97, 105, 248


def _hazard_at(column: int, **state: int) -> MemoryState:
    bullet = make_row(BULLET, FLAG_LISTED, (column * GRID_CELL_PX, 200, column * GRID_CELL_PX + 7, 207))
    return make_state(has_player=0, instances=make_rows(bullet), **state)


def _hazard_columns(grid: np.ndarray, steps_ago: int) -> list[int]:
    channel = len(Channel) + (steps_ago - 1) * len(MOTION_CHANNELS) + MOTION_CHANNELS.index(Channel.HAZARD) if steps_ago else Channel.HAZARD
    return np.flatnonzero(grid[channel].any(axis=0)).tolist()


SNAKE, SHOPKEEPER, SKELETON = 111, 284, 120  # stomp rules: yes, no, falling


def test_a_wall_box_fills_exactly_the_cells_it_covers():
    wall = make_row(57, FLAG_SOLID, (16, 32, 31, 47))

    grid = encode_grid(make_state(has_player=0, instances=make_rows(wall)))

    covered = np.argwhere(grid[Channel.SOLID])
    rows = {int(row) for row, _ in covered}
    columns = {int(column) for _, column in covered}
    assert rows == {32 // GRID_CELL_PX, 47 // GRID_CELL_PX}
    assert columns == {16 // GRID_CELL_PX, 31 // GRID_CELL_PX}


def test_instances_are_placed_relative_to_the_view():
    wall = make_row(57, FLAG_SOLID, (160, 800, 167, 807))

    grid = encode_grid(make_state(has_player=0, view=(160.0, 800.0, 160.0, 284.0), instances=make_rows(wall)))

    assert grid[Channel.SOLID, 0, 0] == 1
    assert grid[Channel.SOLID].sum() == 1


def test_no_view_encodes_an_empty_grid():
    wall = make_row(57, FLAG_SOLID, (0, 0, 7, 7))

    grid = encode_grid(make_state(view=(0.0, 0.0, 0.0, 0.0), instances=make_rows(wall)))

    assert not grid.any()


def test_instances_outside_the_view_are_dropped_like_on_screen():
    far = make_row(57, FLAG_SOLID, (5000, 5000, 5100, 5100))

    grid = encode_grid(make_state(has_player=0, instances=make_rows(far)))

    assert not grid.any()


def test_enemies_split_by_what_touching_them_would_do():
    # player at the top, standing; every enemy below it
    snake = make_row(SNAKE, FLAG_ENEMY, (0, 100, 7, 107))
    skeleton = make_row(SKELETON, FLAG_ENEMY, (40, 100, 47, 107))
    shopkeeper = make_row(SHOPKEEPER, FLAG_ENEMY, (80, 100, 87, 107))

    grid = encode_grid(make_state(ypos=20.0, grounded=1, instances=make_rows(snake, skeleton, shopkeeper)))

    assert grid[Channel.STOMP_NOW, 100 // GRID_CELL_PX, 0] == 1
    assert grid[Channel.STOMP_FROM_ABOVE, 100 // GRID_CELL_PX, 40 // GRID_CELL_PX] == 1  # needs the player falling
    assert grid[Channel.HURTS, 100 // GRID_CELL_PX, 80 // GRID_CELL_PX] == 1


def test_spikes_hazards_and_gems_get_their_own_channels():
    spike = make_row(SPIKE, FLAG_LISTED, (0, 200, 7, 207))
    bullet = make_row(BULLET, FLAG_LISTED, (40, 200, 47, 207))
    gem = make_row(GEM, FLAG_LISTED, (80, 200, 87, 207))

    grid = encode_grid(make_state(has_player=0, instances=make_rows(spike, bullet, gem)))

    assert grid[Channel.SPIKES].sum() == 1
    assert grid[Channel.HAZARD].sum() == 1
    assert grid[Channel.GEM].sum() == 1


def test_the_player_is_drawn_where_it_stands():
    grid = encode_grid(make_state(xpos=80.0, ypos=100.0))

    assert grid[Channel.PLAYER, 100 // GRID_CELL_PX, 80 // GRID_CELL_PX] == 1


def test_zero_max_hp_and_ammo_do_not_divide_by_zero():
    vec = encode_vector(make_state(hp=0, hp_max=0, ammo=0, ammo_max=0))

    assert np.isfinite(vec).all()


def test_menu_encodes_the_cursor_and_each_offer_and_skips_empty_or_placeholder_offers():
    vec = encode_vector(make_state(has_player=0, menu=1, cursor=2, offers=[13, -1, 100, 6]))
    menu = vec[VEC_MENU]
    offers = menu[1 + OFFER_SLOTS :].reshape(OFFER_SLOTS, N_UPGRADES)

    assert menu[0] == 1.0
    assert menu[1 : 1 + OFFER_SLOTS].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert offers.sum(axis=1).tolist() == [1.0, 0.0, 0.0, 1.0]
    assert offers[0, 13] == 1.0


def test_menu_section_is_empty_outside_the_menu():
    vec = encode_vector(make_state(cursor=1, offers=[13, 6, 9, -1]))

    assert not vec[VEC_MENU].any()


def test_shop_encodes_its_cursor_and_prices():
    vec = encode_vector(make_state(shop=1, shop_cursor=1, shop_prices=[0, 150, 500, 1000]))
    shop = vec[VEC_SHOP]

    assert shop[0] == 1.0
    assert shop[1 : 1 + SHOP_SLOTS].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert shop[1 + SHOP_SLOTS + 2] == 0.5


def test_render_draws_the_menu_at_the_same_size_as_play():
    menu = make_state(has_player=0, menu=1, cursor=1, offers=[13, 6, 9, -1])
    play = make_state()

    assert render_obs(ObservationEncoder().encode(menu)).shape == render_obs(ObservationEncoder().encode(play)).shape


def test_grid_matches_the_declared_shape():
    assert encode_grid(make_state()).shape == FRAME_SHAPE
    assert ObservationEncoder().encode(make_state())["grid"].shape == GRID_SHAPE


def test_the_past_frames_show_where_a_moving_hazard_was():
    encoder = ObservationEncoder()
    for column in (0, 1):
        encoder.encode(_hazard_at(column))

    grid = encoder.encode(_hazard_at(2))["grid"]

    assert [_hazard_columns(grid, steps_ago) for steps_ago in (0, 1, 2)] == [[2], [1], [0]]


def test_the_first_frame_repeats_the_present_as_its_past():
    grid = ObservationEncoder().encode(_hazard_at(3))["grid"]

    assert [_hazard_columns(grid, steps_ago) for steps_ago in (0, 1, 2)] == [[3], [3], [3]]


def test_a_new_level_forgets_the_previous_level_frames():
    encoder = ObservationEncoder()
    encoder.encode(_hazard_at(0, level=1))

    grid = encoder.encode(_hazard_at(5, level=2))["grid"]

    assert _hazard_columns(grid, 1) == [5]
