from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from model import DuelingNetwork
from utils import ExperienceReplayMemory, AgentConfig, BranchingTensorEnv
import utils

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class BranchingDQN(nn.Module):
    def __init__(self, obs, ac, config):
        super().__init__()

        self.q = DuelingNetwork(obs, ac)
        self.target = DuelingNetwork(obs, ac)

        self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0

    def get_action(self, x):
        with torch.no_grad():
            # a = self.q(x).max(1)[1]
            out = self.q(x)
            action = torch.argmax(out)
        return action.item()

    def update_policy(self, adam, memory, params):
        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0], -1)
        rewards = torch.tensor(b_rewards).float().reshape(-1, 1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1, 1)

        if torch.cuda.is_available():
            states = states.to('cuda')
            actions = actions.to('cuda')
            next_states = next_states.to('cuda')
            rewards = rewards.to('cuda')
            masks = masks.to('cuda')

        qvals = self.q(states)
        current_q_values = self.q(states).gather(1, actions).squeeze(-1)

        with torch.no_grad():
            argmax = torch.argmax(self.q(next_states), dim=1)

            max_next_q_vals = self.target(next_states).gather(1, argmax.unsqueeze(1))
            max_next_q_vals = max_next_q_vals.mean(1, keepdim=True)

        expected_q_vals = rewards + max_next_q_vals * 0.99 * masks # Belmann
        loss = F.mse_loss(expected_q_vals, current_q_values)
        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())

def train():
    args = utils.arguments()

    bins = 6
    env = BranchingTensorEnv(args.env)

    config = AgentConfig()
    memory = ExperienceReplayMemory(config.memory_size)
    agent = BranchingDQN(env.observation_space.shape[0], env.action_space.n, config)
    adam = optim.Adam(agent.q.parameters(), lr=config.lr)

    s = env.reset()
    ep_reward = 0.
    recap = []

    p_bar = tqdm(total=config.max_frames)
    for frame in range(config.max_frames):
        epsilon = config.epsilon_by_frame(frame)

        if np.random.random() > epsilon:
            action = agent.get_action(s)
        else:
            action = np.random.randint(0, env.action_space.n)

        ns, r, done, infos = env.step(action)
        ep_reward += r

        if done:
            ns = env.reset()
            recap.append(ep_reward)
            p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            ep_reward = 0.

        memory.push((s.reshape(-1).numpy().tolist(), action, r, ns.reshape(-1).numpy().tolist(), 0. if done else 1.))
        s = ns
        p_bar.update(1)

        if frame > config.learning_starts:
            agent.update_policy(adam, memory, config)

        if frame % 1000 == 0:
            utils.save(agent, recap, args)

    p_bar.close()

if __name__ == "__main__":
    train()
