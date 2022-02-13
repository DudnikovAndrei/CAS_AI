import logging
from typing import List, Any

import gym
import numpy as np

from experience import Experience

_logger = logging.getLogger(__file__)


class ExperienceGenerator:

    def __init__(self, env: gym.Env):
        self.env = env
        self.actions = self.env.action_space.n
        pass

    def get_experience(self, number: int) -> List[Experience]:
        experiences: List[Any] = []
        while len(experiences) < number:
            _logger.debug("new agent was born")
            done = False
            state = self.env.reset()
            while not done:
                action = np.random.randint(self.actions)
                next_state, reward, done, _ = self.env.step(action)
                experiences.append(
                    Experience(state=state, action=action, reward=reward, next_state=next_state, done=done))
                state = next_state
                if len(experiences) >= number:
                    break
        return experiences
