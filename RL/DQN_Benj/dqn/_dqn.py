import multiprocessing
from collections import OrderedDict
from typing import List, Tuple

import gym
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import DistributedType
from torch import Tensor
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from dataset import RLDataset
from replay_buffer import ReplayBuffer


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, agent: Agent, model: nn.Module,
                 target_model: nn.Module, torch_device: torch.device, summary_writer: SummaryWriter,
                 batch_size: int = 16, lr: float = 1e-2,
                 gamma: float = 0.99, sync_rate: int = 10, eps_last_frame: int = 1000, eps_start: float = 1.0,
                 eps_end: float = 0.01, episode_length: int = 200):
        """
        :param env: gym environment
        :param replay_buffer: replay buffer containing experiences
        :param agent: agent which plays environment
        :param model: model
        :param target_model: model that gets trained
        :param torch_device: torch device
        :param batch_size: batch size
        :param lr: learning rate
        :param gamma:
        :param sync_rate: how often model and target_model are synced
        :param eps_last_frame:
        :param eps_start:
        :param eps_end:
        :param episode_length:
        """
        super().__init__()

        self.env = env
        self.buffer = replay_buffer
        self.agent = agent
        self.model = model
        self.target_model = target_model
        self.torch_device = torch_device

        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.sync_rate = sync_rate
        self.eps_last_frame = eps_last_frame
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.episode_length = episode_length

        self.total_reward = 0
        self.episode_reward = 0

        self._summary_writer: SummaryWriter = summary_writer

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.model(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.model(states).gather(1, actions.type(torch.int64).unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards
        loss_function = nn.MSELoss()
        return loss_function(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        epsilon = max(self.eps_end, self.eps_start - self.global_step + 1 / self.eps_last_frame)

        # step through environment with agent
        reward, done, experience = self.agent.play_step(self.model, self.torch_device, epsilon)
        self.buffer.append(experience)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        total_reward = torch.tensor(self.total_reward).to(self.torch_device)
        reward = torch.tensor(reward).to(self.torch_device)
        log = {"total_reward": total_reward,
               "reward": reward,
               "train_loss": loss}
        status = {"steps": torch.tensor(self.global_step).to(self.torch_device),
                  "total_reward": torch.tensor(self.total_reward).to(self.torch_device)}

        self._summary_writer.add_scalar("reward/total", total_reward, self.global_step)
        self._summary_writer.add_scalar("reward/current", reward, self.global_step)
        self._summary_writer.add_scalar("loss", loss, self.global_step)

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        num_cpus = multiprocessing.cpu_count()/2
        num_cpus=1
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=int(num_cpus))
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()
