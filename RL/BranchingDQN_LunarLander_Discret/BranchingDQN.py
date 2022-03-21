
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from branching_dqn import DuelingNetwork
from utils import ExperienceReplayMemory

class BranchingDQN():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Q-Network
        self.q_local = DuelingNetwork(state_size, action_size)
        self.q_target = DuelingNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=self.config.lr)

        # Replay memory
        self.memory = ExperienceReplayMemory(action_size, self.config.memory_size, self.config.batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.gamma)

    def get_action(self, state, action_space_n, epsilon=0.):
        if np.random.random() > epsilon:
            with torch.no_grad():
                out = self.q_local(state).unsqueeze(0)
                action = torch.argmax(out)
            return action.numpy()
        else:
            return np.random.randint(0, action_space_n)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)

        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)

        ### Calculate expected value from local network
        q_expected = self.q_local(states).gather(1, actions)

        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_local, self.q_target, self.config.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target"""

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)