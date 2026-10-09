from typing import Final

import numpy as np

from downwell_ai.env._game_data import STOMP_RULE
from downwell_ai.env._game_data import Stomp
from downwell_ai.env.memory import FLAG_ENEMY
from downwell_ai.env.memory import RECORD_INDEX
from downwell_ai.env.memory import MemoryState


_BELOW_MARGIN: Final[float] = 2.0
_LANDING_OFFSET: Final[float] = 8.0
_IS_SET: Final[float] = 0.5

_RULE_CODES: Final[dict[Stomp, int]] = {rule: code for code, rule in enumerate(Stomp)}
_NO_CODE: Final[int] = _RULE_CODES[Stomp.NO]


# lookup table since this runs for every instance on every step
_CODE_BY_OBJECT: Final[np.ndarray] = np.full(max(STOMP_RULE) + 1, _NO_CODE, dtype=np.int8)
for _object, _rule in STOMP_RULE.items():
    _CODE_BY_OBJECT[_object] = _RULE_CODES[_rule]


def _rule_codes(objects: np.ndarray) -> np.ndarray:
    ids = objects.astype(np.int32)
    is_known = (ids >= 0) & (ids < len(_CODE_BY_OBJECT))
    return np.where(is_known, _CODE_BY_OBJECT[np.where(is_known, ids, 0)], _NO_CODE)


def stomp_flags(state: MemoryState) -> tuple[np.ndarray, np.ndarray]:
    """Per instance row, whether contact now stomps it and whether landing on it from above would."""
    rows = state.instances
    if len(rows) == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty

    is_enemy = (rows[:, RECORD_INDEX["flags"]].astype(np.int32) & FLAG_ENEMY) != 0
    codes = np.where(is_enemy, _rule_codes(rows[:, RECORD_INDEX["object"]]), _NO_CODE)

    yy = rows[:, RECORD_INDEX["yy"]]
    y = rows[:, RECORD_INDEX["y"]]
    white = rows[:, RECORD_INDEX["white"]] > _IS_SET
    angry = rows[:, RECORD_INDEX["angry"]] > _IS_SET

    below_now = yy > state.ypos + _BELOW_MARGIN
    below_landing = yy > y - _LANDING_OFFSET + _BELOW_MARGIN
    is_falling = state.grounded == 0 and state.vy > 0

    def by_rule(below: np.ndarray, falling: bool) -> np.ndarray:  # noqa: FBT001 (mirrors the handlers' own flag)
        return np.select(
            [
                codes == _RULE_CODES[Stomp.YES],
                codes == _RULE_CODES[Stomp.FALLING],
                codes == _RULE_CODES[Stomp.WHEN_WHITE],
                codes == _RULE_CODES[Stomp.WHEN_CALM],
            ],
            [below, below & falling, below & white, below & ~angry],
            default=False,
        )

    now = by_rule(below_now, is_falling)
    # from above means landing while falling, with the player just over the enemy
    from_above = by_rule(below_landing, True)  # noqa: FBT003 (landing is by definition falling)

    return now, from_above
