import numpy as np

from downwell_ai.env.memory import N_UPGRADES
from downwell_ai.env.memory import OFFER_SLOTS
from downwell_ai.env.memory import RECORD_FIELDS
from downwell_ai.env.memory import RECORD_INDEX
from downwell_ai.env.memory import SHOP_SLOTS
from downwell_ai.env.memory import parse_snapshot
from downwell_ai.env.memory.layout import HEADER_FIELDS
from downwell_ai.env.memory.layout import HEADER_INDEX


def _snapshot(records: list[dict[str, float]] | None = None, offers: list[float] | None = None, **header: float) -> bytes:
    """Pack a snapshot buffer the way the patch writes it."""
    values = np.zeros(len(HEADER_FIELDS), dtype="<f4")
    for name, value in header.items():
        values[HEADER_INDEX[name]] = value
    offer_values = np.array(offers or [-1.0] * OFFER_SLOTS, dtype="<f4")
    prices = np.zeros(SHOP_SLOTS, dtype="<f4")
    owned = np.zeros(N_UPGRADES, dtype="<f4")
    rows = np.zeros((len(records or []), len(RECORD_FIELDS)), dtype="<f4")
    for index, record in enumerate(records or []):
        for name, value in record.items():
            rows[index, RECORD_INDEX[name]] = value
    count = np.array([len(rows)], dtype="<f4")
    return b"".join(part.tobytes() for part in (values, offer_values, prices, owned, count, rows))


def test_header_fields_land_in_their_state_fields():
    state = parse_snapshot(_snapshot(hp=3, hp_max=4, gems=120, combo=5, xpos=40.5, view_x=160, view_y=800, view_w=160, view_h=284))

    assert (state.hp, state.hp_max, state.gems, state.combo) == (3, 4, 120, 5)
    assert state.xpos == 40.5
    assert state.view == (160.0, 800.0, 160.0, 284.0)


def test_records_become_one_row_each_in_order():
    state = parse_snapshot(_snapshot(records=[{"object": 111, "x": 10}, {"object": 97, "x": 20}]))

    assert state.instances.shape == (2, len(RECORD_FIELDS))
    assert state.instances[:, RECORD_INDEX["object"]].tolist() == [111, 97]
    assert state.instances[1, RECORD_INDEX["x"]] == 20


def test_a_snapshot_without_records_has_an_empty_instance_array():
    state = parse_snapshot(_snapshot())

    assert state.instances.shape == (0, len(RECORD_FIELDS))


def test_offers_keep_their_slots_including_empty_ones():
    state = parse_snapshot(_snapshot(offers=[13, -1, 100, 6], menu=1, cursor=2))

    assert state.offers == [13, -1, 100, 6]
    assert (state.menu, state.cursor) == (1, 2)
