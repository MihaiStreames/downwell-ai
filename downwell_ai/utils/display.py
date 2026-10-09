from types import TracebackType
from typing import Self

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import Task
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from downwell_ai.metrics import EpisodeMetrics
from downwell_ai.metrics import LearnMetrics
from downwell_ai.metrics import Metrics


console = Console()


def build_episode_table(episodes: list[EpisodeMetrics], title: str, index_label: str = "step") -> Table:
    """One row per episode with the best return highlighted; `index_label` names what each episode's `step` holds."""
    table = Table(title=title)
    table.add_column(index_label, justify="right")
    table.add_column("return", justify="right")
    table.add_column("level", justify="right")
    table.add_column("depth", justify="right")
    table.add_column("outcome")

    best_return = max(episode.episode_return for episode in episodes)

    for episode in episodes:
        table.add_row(
            f"{episode.step:,}",
            f"{episode.episode_return:.1f}",
            f"{episode.area}-{episode.level}",
            f"{episode.depth:.0f}",
            "terminated" if episode.is_terminated else "truncated",
            style="bold green" if episode.episode_return == best_return else None,
        )

    return table


class _StepsPerSecondColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(f"{task.speed:,.0f} steps/s" if task.speed else "- steps/s", style="magenta")


class TrainingDisplay:
    """Live progress bar during training, evaluation episode table once it ends."""

    def __init__(self, total_steps: int) -> None:
        self.__progress = Progress(
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("return [cyan]{task.fields[episode_return]:>6.1f}[/]"),
            TextColumn("eval [green]{task.fields[evaluation_return]:>6.1f}[/]"),
            TextColumn("level {task.fields[level]}"),
            TextColumn("eps {task.fields[eps]:.2f}"),
            _StepsPerSecondColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        self.__task = self.__progress.add_task("train", total=total_steps, episode_return=0.0, evaluation_return=0.0, level="1-1", eps=1.0)
        self.__evaluation_episodes: list[EpisodeMetrics] = []

    def update(self, metrics: Metrics) -> None:
        """Refresh the progress bar from a new metrics record."""
        if isinstance(metrics, LearnMetrics):
            return

        if metrics.is_evaluation:
            self.__evaluation_episodes.append(metrics)
            self.__progress.update(self.__task, completed=metrics.step, evaluation_return=metrics.episode_return)
            return

        self.__progress.update(
            self.__task, completed=metrics.step, episode_return=metrics.episode_return, level=f"{metrics.area}-{metrics.level}", eps=metrics.eps
        )

    def __enter__(self) -> Self:  # noqa: D105
        self.__progress.start()
        return self

    def __exit__(  # noqa: D105
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__progress.stop()
        if exception is None and self.__evaluation_episodes:
            console.print(build_episode_table(self.__evaluation_episodes, "evaluation episodes"))
