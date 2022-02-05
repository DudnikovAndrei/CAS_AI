import numpy as np
import os
os.system('python -m tensorflow.tensorboard --logdir=/logging_util/logs')

import matplotlib.pyplot as plt

from IPython import display

# Set plotting options
try:
    get_ipython().magic("matplotlib inline")
except:
    plt.ion()

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
plt.interactive(True)

import time



import datetime
import logging

import gym
import torch
import torch.nn
from pytorch_lightning import Trainer
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from config import current_config
from dqn import DQNLightning
from experience_generator import AgentBasedExperienceGenerator
from logging_util.builder import LoggingBuilder
from model import Model
from replay_buffer import ReplayBuffer

_logger = logging.getLogger()
builder = LoggingBuilder(_logger)
builder.add_default_loggers(__file__)


def visualize_agent(env, model):
    state = model.agent.env.reset()
    img = plt.imshow(model.agent.env.render(mode='rgb_array'))
    for t in range(1000):
        # step through environment with agent
        action, _ = model.agent.play_step(model.net, 0, 'cpu')
        # action = env.action_space.sample()
        img.set_data(model.agent.env.render(mode='rgb_array'))
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, reward, done, _ = model.agent.env.step(int(action))
        if done:
            print('Score: ', t + 1)
            break

    env.close()


def main():
    try:
        _logger.info("main started")

        _logger.info(f"cuda available: {torch.cuda.is_available()}")

        experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = gym.make("CartPole-v0")

        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        model = Model(obs_size, n_actions)
        target_model = Model(obs_size, n_actions)

        agent = Agent(env)

        experience_generator = AgentBasedExperienceGenerator(agent=agent, torch_device=torch_device, model=model)

        initial_experience = experience_generator.generate_experience(number=1000)

        replay_buffer = ReplayBuffer(1000) # 1000
        replay_buffer.extend(initial_experience)

        summary_writer = SummaryWriter(log_dir=str(current_config.tensorboard_path.joinpath(experiment_name)))

        dqn = DQNLightning(env=env, replay_buffer=replay_buffer, agent=agent, model=model, target_model=target_model,
                           torch_device=torch_device, summary_writer=summary_writer)

        trainer = Trainer(gpus=min(1, torch.cuda.device_count()),
                          max_epochs=100, # 200
                          val_check_interval=100)

        trainer.fit(dqn)

        visualize_agent(env, model)

    except Exception as e:
        _logger.exception(f"exception occurred {e}", exc_info=True)

    finally:
        _logger.info("main finished")


if __name__ == "__main__":
    main()
