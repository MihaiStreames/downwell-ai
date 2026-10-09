from collections.abc import Callable
import math
from pathlib import Path
from typing import Final

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from downwell_ai.metrics._records import EpisodeMetrics
from downwell_ai.metrics._records import LearnMetrics
from downwell_ai.metrics._records import Metrics


_LEVELS_PER_AREA: Final[int] = 3

_ROLLING_POINTS: Final[int] = 40
_LEARN_BINS: Final[int] = 300
_COLUMNS: Final[int] = 3

_TRAINING_RETURN: Final[str] = "training return"
_EVALUATION_RETURN: Final[str] = "evaluation return"
_DEPTH: Final[str] = "depth (px)"
_LEVELS: Final[str] = "levels cleared"
_HITS: Final[str] = "hits per episode"
_DETOURS: Final[str] = "detours per episode"
_LENGTH: Final[str] = "episode length (steps)"
_LOSS: Final[str] = "loss"
_MEAN_Q: Final[str] = "mean Q"
_EPS: Final[str] = "epsilon"
_COMPONENTS: Final[str] = "reward components"


def _levels_cleared(episode: EpisodeMetrics) -> float:
    return (episode.area - 1) * _LEVELS_PER_AREA + episode.level - 1


def _plot_rolling(
    axes: Axes, episodes: list[EpisodeMetrics], value: Callable[[EpisodeMetrics], float], label: str, *, color: str | None = None, linestyle: str = "-"
) -> None:
    if not episodes:
        return

    window = max(1, len(episodes) // _ROLLING_POINTS)
    steps = np.array([episode.step for episode in episodes])
    values = np.array([value(episode) for episode in episodes], dtype=np.float64)
    means = np.convolve(values, np.ones(window) / window, mode="valid")

    axes.plot(steps[window - 1 :], means, label=label, color=color, linestyle=linestyle)


def _plot_binned(axes: Axes, learning: list[LearnMetrics], value: Callable[[LearnMetrics], float], label: str) -> None:
    if not learning:
        return

    steps = np.array([record.step for record in learning], dtype=np.float64)
    values = np.array([value(record) for record in learning], dtype=np.float64)

    edges = np.linspace(steps[0], steps[-1] + 1, _LEARN_BINS + 1)
    bins = np.digitize(steps, edges) - 1

    counts = np.bincount(bins, minlength=_LEARN_BINS)
    filled = counts > 0

    means = np.bincount(bins, weights=values, minlength=_LEARN_BINS)[filled] / counts[filled]
    centers = ((edges[:-1] + edges[1:]) / 2)[: len(counts)][filled]

    axes.plot(centers, means, label=label)


def _plot_run(axes: dict[str, Axes], name: str, records: list[Metrics], *, is_single: bool) -> None:
    episodes = [record for record in records if isinstance(record, EpisodeMetrics)]
    training = [episode for episode in episodes if not episode.is_evaluation]
    evaluations = [episode for episode in episodes if episode.is_evaluation]
    learning = [record for record in records if isinstance(record, LearnMetrics)]

    _plot_rolling(axes[_TRAINING_RETURN], training, lambda episode: episode.episode_return, name)

    if evaluations:
        axes[_EVALUATION_RETURN].plot([episode.step for episode in evaluations], [episode.episode_return for episode in evaluations], ".-", label=name)

    _plot_rolling(axes[_DEPTH], training, lambda episode: episode.depth, name)
    _plot_rolling(axes[_LEVELS], training, _levels_cleared, name)
    _plot_rolling(axes[_LENGTH], training, lambda episode: episode.episode_length, name)
    _plot_rolling(axes[_EPS], training, lambda episode: episode.eps, name)
    _plot_binned(axes[_LOSS], learning, lambda record: record.loss, name)
    _plot_binned(axes[_MEAN_Q], learning, lambda record: record.mean_q, name)

    has_hits = any(episode.hits for episode in training)
    has_detours = any(episode.side_rooms or episode.shops or episode.upgrades for episode in training)

    if not is_single:
        if has_hits:
            _plot_rolling(axes[_HITS], training, lambda episode: sum(episode.hits.values()), name)
        if has_detours:
            _plot_rolling(axes[_DETOURS], training, lambda episode: episode.side_rooms, name)
        return

    if has_hits:
        _plot_rolling(axes[_HITS], training, lambda episode: sum(episode.hits.values()), "total", color="black")

    if has_detours:
        _plot_rolling(axes[_DETOURS], training, lambda episode: episode.side_rooms, "side rooms")
        _plot_rolling(axes[_DETOURS], training, lambda episode: episode.shops, "shops")
        _plot_rolling(axes[_DETOURS], training, lambda episode: episode.purchases, "purchases")

    components = sorted({component for episode in training for component in episode.components})
    for component in components:
        _plot_rolling(axes[_COMPONENTS], training, lambda episode, component=component: episode.components.get(component, 0.0), component)


def plot_runs(runs: dict[str, list[Metrics]], path: Path) -> None:
    """Draw the training curves of one or more runs into one png, `runs` mapping each run's name to its records."""
    is_single = len(runs) == 1
    titles = [_TRAINING_RETURN, _EVALUATION_RETURN, _DEPTH, _LEVELS, _HITS, _DETOURS, _LENGTH, _LOSS, _MEAN_Q, _EPS]
    if is_single:
        titles.append(_COMPONENTS)

    rows = math.ceil(len(titles) / _COLUMNS)
    figure = Figure(figsize=(6 * _COLUMNS, 3.5 * rows), layout="constrained")
    panels = figure.subplots(rows, _COLUMNS, squeeze=False).ravel()
    axes = dict(zip(titles, panels, strict=False))

    for unused in panels[len(titles) :]:
        unused.set_visible(False)

    for name, records in runs.items():
        _plot_run(axes, name, records, is_single=is_single)

    for title, panel in axes.items():
        panel.set_title(title)
        panel.set_xlabel("step")
        panel.grid(visible=True, alpha=0.3)

        if not panel.get_lines():
            panel.text(0.5, 0.5, "not recorded", transform=panel.transAxes, ha="center", va="center", color="gray")

        if len(panel.get_lines()) > 1:
            panel.legend(fontsize="small")

    figure.suptitle(next(iter(runs)) if is_single else "comparison")
    figure.savefig(path, dpi=100)
