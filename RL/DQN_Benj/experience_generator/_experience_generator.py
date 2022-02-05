from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from agent import Agent
from experience import Experience


class IExperienceGenerator(ABC):

    @abstractmethod
    def generate_experience(self, number: int) -> List[Experience]:
        """Generatates number occurences of Experience."""


class AgentBasedExperienceGenerator(IExperienceGenerator):
    _agent: Agent
    _torch_device: torch.device

    def __init__(self, agent: Agent, torch_device: torch.device, model: nn.Module):
        self._agent = agent
        self._torch_device = torch_device
        self._model = model

    def generate_experience(self, number: int) -> List[Experience]:
        """
        Carries out several random steps through the environment to generate experience.
        """
        experiences: List[Experience] = []
        for index in range(number):
            reward, done, experience = self._agent.play_step(self._model, epsilon=1.0, device=self._torch_device)
            experiences.append(experience)
        return experiences
