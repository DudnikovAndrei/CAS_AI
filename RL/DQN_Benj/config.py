from pathlib import Path


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


current_config = Config(tensorboard_path=Path.cwd().joinpath("tensorboard_logs"),
                        models_path=Path.cwd().joinpath("models"))
