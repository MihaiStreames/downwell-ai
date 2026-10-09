from typing import Final


### Process constants ###

PROCESS_NAME: Final[str] = "Downwell.exe"

### Actions ###

ACTION_NONE: Final[int] = 0
ACTION_JUMP: Final[int] = 1
ACTION_LEFT: Final[int] = 2
ACTION_RIGHT: Final[int] = 3
ACTION_LEFT_JUMP: Final[int] = 4
ACTION_RIGHT_JUMP: Final[int] = 5

# held buttons written to dqn_act, the game derives pressed/released edges itself
INPUT_LEFT: Final[int] = 1
INPUT_RIGHT: Final[int] = 2
INPUT_JUMP: Final[int] = 4

ACTION_INPUTS: Final[dict[int, int]] = {
    ACTION_NONE: 0,
    ACTION_JUMP: INPUT_JUMP,
    ACTION_LEFT: INPUT_LEFT,
    ACTION_RIGHT: INPUT_RIGHT,
    ACTION_LEFT_JUMP: INPUT_LEFT | INPUT_JUMP,
    ACTION_RIGHT_JUMP: INPUT_RIGHT | INPUT_JUMP,
}
N_ACTIONS: Final[int] = len(ACTION_INPUTS)

### Timing ###

GAME_FPS: Final[int] = 60  # native room_speed, game time is measured in these frames
