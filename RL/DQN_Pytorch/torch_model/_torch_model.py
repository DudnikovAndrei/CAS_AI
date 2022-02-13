import numpy as np
import torch
from torch import Tensor
from torch import nn


class TorchModel(nn.Module):
    _n_states: int
    _n_actions: int
    _hidden_size: int

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 24):
        super().__init__()
        self._n_states = n_states
        self._n_actions = n_actions
        self._hidden_size = hidden_size

        self._internal_network = nn.Sequential(
            nn.Linear(self._n_states, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._n_actions))

    def forward(self, x: Tensor):
        logits = self._internal_network(x)
        return logits

    def __call__(self, x: np.ndarray):
        x = torch.from_numpy(x)
        return self.forward(x)
