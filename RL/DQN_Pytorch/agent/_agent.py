import collections
import random
from typing import Deque, List

import gym
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from experience import Experience, experiences_to_numpy
from model_repository import ModelRepository


class Agent:
    def __init__(self, env: gym.Env,
                 model: nn.Module,
                 target_model: nn.Module,
                 optimizer: Optimizer,
                 loss_function: MSELoss,
                 memory: Deque,
                 model_repository: ModelRepository,
                 summary_writer: SummaryWriter,
                 min_memory_size: int = 1000):

        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.memory = memory
        self.model_repository = model_repository
        self.summary_writer = summary_writer

        self.actions = self.env.action_space.n

        self.min_memory_size = min_memory_size

        if len(self.memory) < self.min_memory_size:
            raise ValueError("not enough samples in memory, please fill memory first")

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.batch_size = 32
        self._global_step = 0

    def get_action(self, state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.model(state).detach().numpy())

    def train(self, n_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=5)
        best_reward_mean = 0.0

        for episode in range(1, n_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            current_time_step_in_environment = 0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                current_time_step_in_environment += 1

                if done and current_time_step_in_environment < 499:
                    reward = -100.0

                experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)

                self.memory.append(experience)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                self.replay()
                total_reward += reward

                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                state = next_state

                self._global_step += 1

                if done:
                    if total_reward < 500:
                        total_reward += 100.0
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    self.summary_writer.add_scalar("reward/episode", total_reward, episode)
                    self.summary_writer.add_scalar("epsilon/current", self.epsilon, episode)

                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)

                    if current_reward_mean > best_reward_mean:
                        model_state = self.model.state_dict()
                        self.target_model.load_state_dict(model_state)
                        best_reward_mean = current_reward_mean

                        self.model_repository.save_model(model=self.model, filename="model")
                        self.model_repository.save_model(model=self.target_model, filename="target_model")

                        print(f"New best mean: {best_reward_mean}")

                        if best_reward_mean > 400:
                            return
                    break

    def replay(self):
        minibatch: List[Experience] = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = experiences_to_numpy(minibatch)

        q_values = self.model(states).detach().numpy()
        with torch.no_grad():
            q_values_next = self.target_model(next_states).detach().numpy()

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        q_values = torch.from_numpy(q_values)

        pred = self.model(states)
        loss = self.loss_function(pred, q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def play(self, num_episodes: int, render: bool = True):
        self.model.load_state_dict(self.model_repository.load_state_dict("model"))
        self.target_model.load_state_dict(self.model_repository.load_state_dict("target_model"))

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)

            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                total_reward += reward
                state = next_state

                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break
