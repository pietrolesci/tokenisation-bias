from dataclasses import dataclass, field

from lightning.pytorch import LightningModule
from lightning_utilities import StrEnum
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, get_scheduler

from src.model import MODEL_TYPE
from src.utilities import DictConfig, get_logger

logger = get_logger("module")

TYPE_TO_OPTIMIZER_CLASS = {"adamw": AdamW}


class RunningStage(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class OptimCofig(DictConfig):
    # Optimizer config
    optim_name: str
    lr: float
    weight_decay: float = 0.0
    optim_kwargs: dict = field(default_factory=dict)

    # Scheduler config
    scheduler_name: str | None = None
    num_warmup_steps: int | None = None
    scheduler_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.optim_name in TYPE_TO_OPTIMIZER_CLASS
        if self.scheduler_name is not None:
            assert self.scheduler_name in TYPE_TO_SCHEDULER_FUNCTION


class LanguageModel(LightningModule):
    def __init__(self, model: MODEL_TYPE, optim_config: OptimCofig) -> None:
        super().__init__()
        self.model = model
        self.optim_config = optim_config
        self.save_hyperparameters(ignore=["model", "model_config"])

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.model.forward(input_ids=input_ids).logits  # type: ignore

    def step(self, batch: dict, stage: RunningStage) -> Tensor | None:
        input_ids = batch["input_ids"][:, :-1]
        labels = batch["input_ids"][:, 1:].clone()
        logits = self.forward(input_ids)
        loss = cross_entropy(logits.permute(0, 2, 1), labels)

        self.log(
            f"{stage}/loss",
            loss.detach(),
            on_step=stage == RunningStage.TRAIN,
            on_epoch=stage == RunningStage.VALIDATION,
            prog_bar=True,
            logger=True,
            batch_size=labels.shape[0],
        )

        if stage == RunningStage.TRAIN:
            return loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.step(batch, RunningStage.TRAIN)  # type: ignore

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.step(batch, RunningStage.VALIDATION)  # type: ignore

    def configure_optimizers(self) -> dict | Optimizer:
        # Get params that require grad
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.optim_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = TYPE_TO_OPTIMIZER_CLASS[self.optim_config.optim_name](
            optim_groups, lr=self.optim_config.lr, **self.optim_config.optim_kwargs
        )
        if self.optim_config.scheduler_name is not None:
            scheduler = get_scheduler(
                name=self.optim_config.scheduler_name,
                num_warmup_steps=self.optim_config.num_warmup_steps,
                optimizer=optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                scheduler_specific_kwargs=self.optim_config.scheduler_kwargs,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        return optimizer
