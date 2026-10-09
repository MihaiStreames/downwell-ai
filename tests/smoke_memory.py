import time

import pytest

from downwell_ai.consts import PROCESS_NAME
from downwell_ai.env.memory import attach


pytestmark = pytest.mark.smoke


def setup_module():
    print("\nfocus the game window, memory read fires in 2s...")
    time.sleep(2)


def test_read_returns_lives():
    state = attach(PROCESS_NAME)
    assert state is not None, f"attaching to {PROCESS_NAME} failed, is the game running?"

    data = state.read()
    assert data.hp > 0, f"hp={data.hp}, start a run first"
    print(
        f"hp={data.hp}/{data.hp_max}  gems={data.gems}  ammo={data.ammo}/{data.ammo_max}  "
        f"pos=({data.xpos:.1f},{data.ypos:.1f})  combo={data.combo}  "
        f"instances={len(data.instances)}  view={data.view}"
    )
