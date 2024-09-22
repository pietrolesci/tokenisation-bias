from copy import copy
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from datatrove.utils.dataset import DatatroveFolderDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.utilities import DictConfig, get_logger, ld_to_dl

logger = get_logger("data")


@dataclass
class DataloaderConfig(DictConfig):
    batch_size: int
    eval_batch_size: int
    num_workers: int | None = cpu_count()
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    multiprocessing_context: str | None = None
    shuffle: bool = False

    def get_train_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs.pop("eval_batch_size")
        return kwargs

    def get_val_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs["batch_size"] = kwargs.pop("eval_batch_size")
        kwargs["shuffle"] = False
        return kwargs


class DataModule(LightningDataModule):
    train_ds: DatatroveFolderDataset
    val_ds: Dataset

    def __init__(
        self,
        train_data_path: str | Path | None,
        val_data_path: str | Path | None,
        max_position_embeddings: int,
        dataloader_config: DataloaderConfig,
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path) if train_data_path else train_data_path
        self.val_data_path = Path(val_data_path) if val_data_path else val_data_path
        self.max_position_embeddings = max_position_embeddings
        self.dataloader_config = dataloader_config
        self.save_hyperparameters()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if self.train_data_path:
            self.train_ds = DatatroveFolderDataset(
                folder_path=str(self.train_data_path),
                filename_pattern=f"{self.train_data_path}/*.ds",
                seq_len=self.max_position_embeddings,
                shuffle=False,
                seed=42,
                token_size=2,
                # token_size=2 if self.config.vocab_size < 65_000 else 4,
            )
            logger.info(f"Train dataset loaded: {len(self.train_ds)=}")
            logger.info(f"{self.train_ds=}")

        if self.val_data_path:
            self.val_ds = load_from_disk(str(self.val_data_path))  # type: ignore
            logger.info(f"Validation dataset loaded: {len(self.val_ds)=}")
            logger.info(f"{self.val_ds=}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, **self.dataloader_config.get_train_kwargs())

    def val_dataloader(self) -> DataLoader:
        max_length = self.max_position_embeddings

        def collator_fn(batch: list[dict[str, int | list[int]]]) -> dict:
            new_batch = ld_to_dl(batch)
            # Truncate
            input_ids: list[list[int]] = [x[:max_length] for x in new_batch["input_ids"]]
            # Pad from the left side
            padded_input_ids = np.vstack(
                [
                    # to understand this function: pad_before == max_length - len(x) and pad_after == 0
                    np.pad(x, (max_length - len(x), 0), mode="constant", constant_values=0)
                    for x in input_ids
                ]
            )
            new_batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)  # type: ignore
            return new_batch

        return DataLoader(self.val_ds, **self.dataloader_config.get_val_kwargs(), collate_fn=collator_fn)  # type: ignore
