import collections
import datetime
import logging
from typing import Deque

import gym
import torch
import torch.nn
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from config import current_config
from experience import Experience
from experience_generator import ExperienceGenerator
from logging_util.builder import LoggingBuilder
from model_repository import ModelRepository
from torch_model import TorchModel

_logger = logging.getLogger()
builder = LoggingBuilder(_logger)
builder.add_default_loggers(__file__)


def main():
    try:
        _logger.info("main started")

        _logger.info(f"cuda available: {torch.cuda.is_available()}")

        experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        experiment_name = "current"

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device("cpu")

        env = gym.make("CartPole-v1")

        n_states = env.env.observation_space.shape[0]
        n_actions = env.env.action_space.n

        learning_rate = 1e-3

        model = TorchModel(n_states=n_states, n_actions=n_actions)
        target_model = TorchModel(n_states=n_states, n_actions=n_actions)

        optimizer = Adam(params=model.parameters(), lr=learning_rate)

        memory: Deque[Experience] = collections.deque(maxlen=100_000) # 100_000

        experience_generator = ExperienceGenerator(env=env)
        experience = experience_generator.get_experience(number=10000)

        memory.extend(experience)

        loss_function = nn.MSELoss()

        model_repository = ModelRepository(basepath=current_config.models_path.joinpath(experiment_name))

        summary_writer = SummaryWriter(log_dir=str(current_config.tensorboard_path.joinpath(experiment_name)))

        agent = Agent(env=env,
                      model=model,
                      target_model=target_model,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      memory=memory,
                      model_repository=model_repository,
                      summary_writer=summary_writer)

        agent.train(n_episodes=1000)
        input("Play?")
        agent.play(num_episodes=20, render=True)

    except Exception as e:
        _logger.exception(f"exception occurred {e}", exc_info=True)

    finally:
        _logger.info("main finished")


if __name__ == "__main__":
    main()
