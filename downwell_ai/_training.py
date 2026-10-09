from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger
import numpy as np
import torch

from downwell_ai.agent import DQNAgent
from downwell_ai.agent import NStepCollector
from downwell_ai.agent import ReplayBuffer
from downwell_ai.config import TrainingConfig
from downwell_ai.env.downwell_env import DownwellEnv
from downwell_ai.metrics import EpisodeMetrics
from downwell_ai.metrics import EpisodeTally
from downwell_ai.metrics import Metrics
from downwell_ai.utils.gif import save_gif
from downwell_ai.utils.schedules import linear_schedule


def _log_gif_failure(future: Future[None]) -> None:
    if (error := future.exception()) is not None:
        logger.error("saving an evaluation gif failed: {}", error)


def play_evaluation_episode(agent: DQNAgent, env: DownwellEnv, step: int, eps: float) -> tuple[EpisodeMetrics, list[np.ndarray]]:
    """Play one evaluation episode, returning its metrics and the rendered frame of every step."""
    obs, info = env.reset()
    tally, frames = EpisodeTally(), []
    is_terminated = is_truncated = False

    while not (is_terminated or is_truncated):
        if (frame := env.render()) is not None:
            frames.append(frame)

        action = agent.select_action(obs, eps)
        obs, reward, is_terminated, is_truncated, info = env.step(action)
        tally.add(reward, info)

    # frames are captured before each step, so add the state the episode ended on
    if (frame := env.render()) is not None:
        frames.append(frame)

    return tally.finish(step, info, is_terminated=is_terminated, eps=eps, is_evaluation=True), frames


def run_training(
    config: TrainingConfig,
    env: DownwellEnv,
    run_directory: Path,
    on_metrics: Callable[[Metrics], None],
    initial_weights: dict[str, torch.Tensor] | None = None,
) -> DQNAgent:
    """Train an agent and return it holding the weights of its best evaluation episode."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("training on {}", device)

    agent = DQNAgent(config, device)
    if initial_weights is not None:
        agent.load_weights(initial_weights)

    buf = ReplayBuffer(config.buf_cap, config.seed, device)
    collector = NStepCollector(buf, config.n_steps, config.gamma)

    eps_start = config.resume_eps_start if initial_weights is not None else config.eps_start
    best_return, best_weights = float("-inf"), agent.copy_weights()

    obs, _ = env.reset(seed=config.seed)
    tally, ep_count = EpisodeTally(), 0

    # gif encoding is slow and unrelated to learning, one background worker keeps saves ordered
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gif-writer") as gif_writer:
        for step in range(1, config.total_steps + 1):
            eps = linear_schedule(eps_start, config.eps_end, config.exploration_steps, step)

            action = agent.select_action(obs, eps)
            next_obs, reward, is_terminated, is_truncated, info = env.step(action)

            collector.add(obs, action, reward, next_obs, is_terminated=is_terminated, is_truncated=is_truncated)
            tally.add(reward, info)
            obs = next_obs

            if step >= config.learning_starts and step % config.train_frequency == 0:
                on_metrics(agent.learn(buf.sample(config.batch_size), step))
            if step % config.target_update_interval == 0:
                agent.sync_target()

            if not (is_terminated or is_truncated):
                continue

            on_metrics(tally.finish(step, info, is_terminated=is_terminated, eps=eps))
            ep_count += 1

            if ep_count % config.eval_interval_episodes == 0:
                evaluation, frames = play_evaluation_episode(agent, env, step, config.eval_eps)
                on_metrics(evaluation)
                gif_writer.submit(save_gif, frames, run_directory / f"eval-{step:08d}.gif", config.frame_skip).add_done_callback(_log_gif_failure)

                if evaluation.episode_return > best_return:
                    best_return, best_weights = evaluation.episode_return, agent.copy_weights()
                    logger.debug("new best evaluation return {:.1f} at step {}", best_return, step)

            obs, _ = env.reset()
            tally = EpisodeTally()

    agent.load_weights(best_weights)
    return agent
