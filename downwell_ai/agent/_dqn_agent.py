from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam

from downwell_ai.agent._network import DQN
from downwell_ai.agent._replay import Batch
from downwell_ai.config import TrainingConfig
from downwell_ai.consts import N_ACTIONS
from downwell_ai.metrics import LearnMetrics


class DQNAgent:
    """Double DQN agent with a hard-synced target network."""

    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        self.__online = DQN().to(device)
        self.__target = deepcopy(self.__online)
        self.__target.train(mode=False)

        self.__optimizer = Adam(self.__online.parameters(), lr=config.learning_rate)

        self.__max_grad_norm = config.max_grad_norm

        self.__device = device
        self.__rng = np.random.default_rng(config.seed)

    def sync_target(self) -> None:
        """Copy the online weights into the target network."""
        self.__target.load_state_dict(self.__online.state_dict())

    def copy_weights(self) -> dict[str, torch.Tensor]:
        """Return a detached copy of the online weights, safe to keep while training continues."""
        return {name: tensor.clone() for name, tensor in self.__online.state_dict().items()}

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load `weights` into both the online and target networks."""
        self.__online.load_state_dict(weights)
        self.__target.load_state_dict(weights)

    def q_values(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return the online network's Q-value for each action."""
        grid = torch.as_tensor(obs["grid"], device=self.__device).unsqueeze(0)
        vec = torch.as_tensor(obs["vec"], device=self.__device).unsqueeze(0)

        with torch.inference_mode():
            return self.__online(grid, vec).squeeze(0).cpu().numpy()

    def select_action(self, obs: dict[str, np.ndarray], eps: float) -> int:
        """Pick a random action with probability `eps`, otherwise the greedy one."""
        if self.__rng.random() < eps:
            return int(self.__rng.integers(N_ACTIONS))
        return int(self.q_values(obs).argmax())

    def learn(self, batch: Batch, step: int) -> LearnMetrics:
        """Take one gradient step towards the double DQN target."""
        with torch.no_grad():
            # online net picks the next action, target net scores it, curbing overestimation
            next_actions = self.__online(batch.next_grids, batch.next_vecs).argmax(dim=1, keepdim=True)
            next_q = self.__target(batch.next_grids, batch.next_vecs).gather(1, next_actions).squeeze(1)
            target_q = batch.rewards + batch.discounts * next_q

        predicted_q = self.__online(batch.grids, batch.vecs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        # huber rather than mse, the unclipped death and level rewards would otherwise dominate gradients
        loss = torch.nn.functional.smooth_l1_loss(predicted_q, target_q)

        self.__optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.__online.parameters(), self.__max_grad_norm)
        self.__optimizer.step()

        return LearnMetrics(step=step, loss=loss.item(), mean_q=predicted_q.mean().item())
