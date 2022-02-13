import logging
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

_logger = logging.getLogger(__file__)


class ModelRepository:

    def __init__(self, basepath: Path):
        self.basepath = basepath

    def save_model(self, model: nn.Module, filename: str):
        model_file = self.basepath.joinpath(f"{filename}.ckpt")
        model_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f=model_file)
        _logger.info(f"saved model to {model_file}")

    def load_state_dict(self, filename: str) -> OrderedDict:
        state_dict = torch.load(self.basepath.joinpath(f"{filename}.ckpt"))
        return state_dict
