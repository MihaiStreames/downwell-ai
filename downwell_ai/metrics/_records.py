import math
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Self

from loguru import logger
from msgspec import DecodeError
from msgspec import Struct
from msgspec import ValidationError
from msgspec.json import Decoder
from msgspec.json import Encoder


class UpgradePick(Struct, frozen=True):
    """One choice in the upgrade menu at the end of a level."""

    area: int
    level: int
    offered: list[str]
    picked: str


class EpisodeMetrics(Struct, frozen=True, tag="episode"):
    """Outcome of one finished episode."""

    step: int
    episode_return: float
    episode_length: int
    area: int
    level: int
    depth: float
    is_terminated: bool
    components: dict[str, float]
    eps: float = math.nan
    is_evaluation: bool = False
    hits: dict[str, int] = {}
    side_rooms: int = 0
    side_room_steps: int = 0
    shops: int = 0
    purchases: int = 0
    gems_spent: int = 0
    upgrades: list[UpgradePick] = []


class LearnMetrics(Struct, frozen=True, tag="learn"):
    """Result of one gradient step."""

    step: int
    loss: float
    mean_q: float


type Metrics = EpisodeMetrics | LearnMetrics


class MetricsWriter:
    """Append metrics to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.__file = path.open("ab")
        self.__encoder = Encoder()

    def write(self, metrics: Metrics) -> None:
        """Append one record as a JSON line."""
        self.__file.write(self.__encoder.encode(metrics) + b"\n")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.__file.close()


class EpisodeTally:
    """Running totals of the episode in progress."""

    def __init__(self) -> None:
        self.__return = 0.0
        self.__length = 0
        self.__components: dict[str, float] = {}
        self.__hits: dict[str, int] = {}
        self.__side_rooms = 0
        self.__side_room_steps = 0
        self.__shops = 0
        self.__purchases = 0
        self.__gems_spent = 0
        self.__upgrades: list[UpgradePick] = []

    def add(self, reward: float, info: dict[str, Any]) -> None:
        """Count one step, `info` being the step's."""
        self.__return += reward
        self.__length += 1

        for name, value in info["reward_components"].items():
            self.__components[name] = self.__components.get(name, 0.0) + value

        if info.get("is_hit"):
            self.__hits["total"] = self.__hits.get("total", 0) + 1

        self.__side_rooms += info.get("entered_side_room", False)
        self.__side_room_steps += info.get("in_side_room", False)
        self.__shops += info.get("entered_shop", False)

        if gems_spent := info.get("gems_spent", 0):
            self.__purchases += 1
            self.__gems_spent += gems_spent

        if (upgrade := info.get("upgrade")) is not None:
            self.__upgrades.append(UpgradePick(**upgrade))

    def finish(self, step: int, info: dict, *, is_terminated: bool, eps: float, is_evaluation: bool = False) -> EpisodeMetrics:
        """Summarize the episode, `info` being the last step's."""
        return EpisodeMetrics(
            step=step,
            episode_return=self.__return,
            episode_length=self.__length,
            area=info["area"],
            level=info["level"],
            depth=info["depth"],
            is_terminated=is_terminated,
            components=dict(self.__components),
            eps=eps,
            is_evaluation=is_evaluation,
            hits=dict(self.__hits),
            side_rooms=self.__side_rooms,
            side_room_steps=self.__side_room_steps,
            shops=self.__shops,
            purchases=self.__purchases,
            gems_spent=self.__gems_spent,
            upgrades=list(self.__upgrades),
        )


def read_metrics(path: Path) -> list[Metrics]:
    """Read every record of a `metrics.jsonl`."""
    decoder = Decoder(Metrics)
    records: list[Metrics] = []
    skipped = 0

    with path.open("rb") as file:
        for line in file:
            try:
                records.append(decoder.decode(line))
            except (ValidationError, DecodeError):
                skipped += 1

    if skipped:
        logger.warning("skipped {} of {} records in {}, written in an older format", skipped, skipped + len(records), path)

    return records
