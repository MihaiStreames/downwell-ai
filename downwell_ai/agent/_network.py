from typing import Final

import torch
from torch import nn

from downwell_ai.consts import N_ACTIONS
from downwell_ai.env.observation import GRID_SHAPE
from downwell_ai.env.observation import VEC_DIM


_HIDDEN_DIM: Final[int] = 128


class DQN(nn.Module):
    """Multi-modal DQN Q-network over the RAM observation."""

    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_SHAPE[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, *GRID_SHAPE)).shape[1]

        self.vec = nn.Sequential(nn.Linear(VEC_DIM, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(cnn_out + 64, _HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(_HIDDEN_DIM, N_ACTIONS),
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape `(B, N_ACTIONS)`."""
        # grid is (B, *GRID_SHAPE) of 0/1; vec is (B, VEC_DIM) float
        z = torch.cat([self.cnn(grid.float()), self.vec(vec)], dim=1)
        return self.head(z)
