import random
from pathlib import Path

import numpy as np
import torch


class Config:
    _tensorboard_path: Path
    _models_path: Path

    def __init__(self, tensorboard_path: Path, models_path: Path):
        self._tensorboard_path = tensorboard_path
        self._models_path = models_path

    @property
    def tensorboard_path(self) -> Path:
        return self._tensorboard_path

    @property
    def models_path(self) -> Path:
        return self._models_path


def seed_random(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


current_config = Config(tensorboard_path=Path.cwd().joinpath("tensorboard_logs"),
                        models_path=Path.cwd().joinpath("model_logs"))

seed_random()
