from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Final

from loguru import logger
import msgspec
from msgspec import DecodeError
from msgspec import ValidationError
import torch
import typer
from typer import BadParameter
from typer import Exit
from typer import Typer

from downwell_ai._training import play_evaluation_episode
from downwell_ai._training import run_training
from downwell_ai.agent import DQNAgent
from downwell_ai.config import TrainingConfig
from downwell_ai.env.downwell_env import DownwellEnv
from downwell_ai.metrics import EpisodeMetrics
from downwell_ai.metrics import Metrics
from downwell_ai.metrics import MetricsWriter
from downwell_ai.metrics import plot_runs
from downwell_ai.metrics import read_metrics
from downwell_ai.utils.display import TrainingDisplay
from downwell_ai.utils.display import build_episode_table
from downwell_ai.utils.display import console
from downwell_ai.utils.exceptions import DownwellError


app = Typer(add_completion=False, pretty_exceptions_show_locals=False)
play_app = Typer(add_completion=False, pretty_exceptions_show_locals=False)
plot_app = Typer(add_completion=False, pretty_exceptions_show_locals=False)

_METRICS_FILE: Final[str] = "metrics.jsonl"
_PLOTS_FILE: Final[str] = "plots.png"


def _create_run_directory(root: Path, config: TrainingConfig) -> Path:
    directory = root / f"downwell-seed{config.seed}-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    directory.mkdir(parents=True)
    (directory / "config.json").write_bytes(msgspec.json.format(msgspec.json.encode(config)))
    return directory


def _load_run_config(model_path: Path) -> TrainingConfig:
    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        message = f"no config.json next to {model_path}, play needs the config the model was trained with"
        raise BadParameter(message, param_hint="MODEL")

    try:
        return msgspec.json.decode(config_path.read_bytes(), type=TrainingConfig)
    except (ValidationError, DecodeError) as error:
        message = f"{config_path}: {error}"
        raise BadParameter(message, param_hint="MODEL") from error


@app.command()
def train(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", exists=True, dir_okay=False, help="training config toml, see configs/README.md")],
    headless: Annotated[bool | None, typer.Option("--headless/--rendered", help="skip drawing while training, overrides the config")] = None,
    resume_path: Annotated[Path | None, typer.Option("--resume", exists=True, dir_okay=False, help="model.pt to keep training from")] = None,
    runs_directory: Path = Path("runs"),
) -> None:
    """Train an agent on the running game, then save its best weights and metrics."""
    try:
        config = msgspec.toml.decode(config_path.read_bytes(), type=TrainingConfig)
    except (ValidationError, DecodeError) as error:
        raise BadParameter(str(error), param_hint="CONFIG") from error

    initial_weights = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path is not None else None

    if headless is not None:
        config = msgspec.structs.replace(config, headless=headless)

    run_directory = _create_run_directory(runs_directory, config)

    # file sink only, the default stderr sink would tear the rich live display
    logger.remove()
    logger.add(run_directory / "training.log", level="DEBUG")
    logger.debug("starting run in {}", run_directory)
    if resume_path is not None:
        logger.debug("resuming from {}", resume_path)

    env = DownwellEnv(config, render_mode="rgb_array")

    try:
        with MetricsWriter(run_directory / _METRICS_FILE) as writer, TrainingDisplay(config.total_steps) as display:

            def record_metrics(metrics: Metrics) -> None:
                writer.write(metrics)
                display.update(metrics)

            agent = run_training(config, env, run_directory, on_metrics=record_metrics, initial_weights=initial_weights)

    except DownwellError as error:
        logger.error("training stopped: {}", error)
        console.print(f"[red]training stopped:[/] {error}")
        raise Exit(1) from error
    finally:
        env.close()

    torch.save(agent.copy_weights(), run_directory / "model.pt")
    plot_runs({run_directory.name: read_metrics(run_directory / _METRICS_FILE)}, run_directory / _PLOTS_FILE)
    console.print(f"run saved to [cyan]{run_directory}[/]")


@play_app.command()
def play(
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="model.pt from a training run, its config.json is read alongside")],
    *,
    episodes: Annotated[int, typer.Option(min=1)] = 3,
    eps: Annotated[float | None, typer.Option(min=0, max=1, help="random action rate, defaults to the run's eval_eps")] = None,
    headless: Annotated[
        bool, typer.Option("--headless/--rendered", help="--rendered plays at real game speed to watch, --headless as fast as possible")
    ] = False,
) -> None:
    """Play a trained model on the running game and summarize each episode."""
    config = msgspec.structs.replace(_load_run_config(model_path), headless=headless)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(config, device)
    agent.load_weights(torch.load(model_path, map_location=device, weights_only=True))

    env = DownwellEnv(config, is_realtime=not headless)
    results: list[EpisodeMetrics] = []

    try:
        for episode in range(1, episodes + 1):
            metrics, _ = play_evaluation_episode(agent, env, step=episode, eps=config.eval_eps if eps is None else eps)
            results.append(metrics)
            console.print(f"episode {episode}: return {metrics.episode_return:.1f}, reached {metrics.area}-{metrics.level}")

    except DownwellError as error:
        console.print(f"[red]play stopped:[/] {error}")
        raise Exit(1) from error
    finally:
        env.close()

    console.print(build_episode_table(results, f"{model_path} ({episodes} episodes)", index_label="episode"))


@plot_app.command()
def plot(
    run_directories: Annotated[list[Path], typer.Argument(metavar="RUN...", exists=True, file_okay=False, help="training run folders from runs/")],
    output_path: Annotated[
        Path | None, typer.Option("--out", dir_okay=False, help="png to write, plots.png in the run or comparison.png beside the runs")
    ] = None,
) -> None:
    """Plot one run's training curves, or overlay several runs to compare them."""
    missing = [directory for directory in run_directories if not (directory / _METRICS_FILE).exists()]
    if missing:
        message = f"no {_METRICS_FILE} in {', '.join(map(str, missing))}"
        raise BadParameter(message, param_hint="RUN")

    if output_path is None:
        output_path = run_directories[0] / _PLOTS_FILE if len(run_directories) == 1 else run_directories[0].parent / "comparison.png"

    plot_runs({directory.name: read_metrics(directory / _METRICS_FILE) for directory in run_directories}, output_path)
    console.print(f"plots saved to [cyan]{output_path}[/]")


if __name__ == "__main__":
    app()
