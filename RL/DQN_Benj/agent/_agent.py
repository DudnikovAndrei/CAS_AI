from typing import Tuple

import gym
import numpy as np
import torch
from torch import nn

from experience import Experience


class Agent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(self, env: gym.Env) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: torch.device) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # state = torch.tensor(np.array([self.state]))
            state = torch.tensor([self.state])

            state = state.cuda(device)

            q_values = net(state)
            # get action with highest q value
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, device: torch.device, epsilon: float = 0.0) -> Tuple[float, bool, Experience]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        experience = Experience(self.state, action, reward, done, new_state)

        self.state = new_state
        if done:
            self.reset()
        return reward, done, experience
