from pathlib import Path

from lightning.pytorch.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
from tbparse import SummaryReader


class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_NAME: str = "tensorboard"

    @property
    def logger_name(self) -> str:
        return self.LOGGER_NAME

    def save_to_parquet(self, path: str | Path) -> None:
        SummaryReader(str(self.log_dir)).scalars.to_parquet(path)
