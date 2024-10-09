from dataclasses import dataclass, field

from lightning.pytorch import LightningModule
from lightning_utilities import StrEnum
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from transformers import PretrainedConfig
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, get_scheduler

from src.model import MODEL_TYPE
from src.optim import HybridLRS, HybridOptim, OrthogonalNesterov
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
    keller_kwargs: dict = field(default_factory=dict)

    # Scheduler config
    scheduler_name: str | None = None
    num_warmup_steps: int | None = None
    scheduler_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.optim_name in TYPE_TO_OPTIMIZER_CLASS
        if self.scheduler_name is not None:
            assert self.scheduler_name in TYPE_TO_SCHEDULER_FUNCTION


class LanguageModel(LightningModule):
    def __init__(self, model: MODEL_TYPE, config: PretrainedConfig, optim_config: OptimCofig) -> None:
        super().__init__()
        self.model = model
        self.config = config  # save it here so that we can find it in the checkpoints!
        self.optim_config = optim_config
        self.save_hyperparameters(ignore=["model"])

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

    def configure_optimizers(self) -> dict:
        # Get params that require grad
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Create optim_groups taking care to check whether we want the keller optimiser
        decay_params, nodecay_params, orthogonal_params = [], [], []
        layer_name = self.optim_config.keller_kwargs.get("layer_name")
        for n, p in param_dict.items():
            if layer_name is not None and p.dim() == 2 and layer_name in n:
                orthogonal_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        optim_groups = [
            {"params": decay_params, "weight_decay": self.optim_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Log some stats
        logger.info(f"{len(decay_params)=}, with {sum(p.numel() for p in decay_params):,} params")
        logger.info(f"{len(nodecay_params)=}, with {sum(p.numel() for p in nodecay_params):,} params")
        if orthogonal_params:
            logger.info(f"{len(orthogonal_params)=}, with {sum(p.numel() for p in orthogonal_params):,} params")

        # Instantiate optimizer(s)
        optimizers = []
        lr_schedulers = []

        # main optimizers
        opt = TYPE_TO_OPTIMIZER_CLASS[self.optim_config.optim_name](
            optim_groups, lr=self.optim_config.lr, **self.optim_config.optim_kwargs
        )
        optimizers.append(opt)

        # maybe add auxiliary keller optimizer
        if orthogonal_params:
            lr_factor = self.optim_config.keller_kwargs["lr_factor"]
            momentum = self.optim_config.keller_kwargs["momentum"]
            orthogonal_nesterov_opt = OrthogonalNesterov(
                orthogonal_params, lr=lr_factor * self.optim_config.lr, momentum=momentum
            )
            optimizers.append(orthogonal_nesterov_opt)

        # Maybe create scheduler
        if self.optim_config.scheduler_name is not None:
            for opt in optimizers:
                scheduler = get_scheduler(
                    name=self.optim_config.scheduler_name,
                    num_warmup_steps=self.optim_config.num_warmup_steps,
                    optimizer=opt,
                    num_training_steps=int(self.trainer.estimated_stepping_batches),
                    scheduler_specific_kwargs=self.optim_config.scheduler_kwargs,
                )
                lr_schedulers.append(scheduler)

        out = {}
        out["optimizer"] = HybridOptim(optimizers) if len(optimizers) > 1 else optimizers[0]
        if len(lr_schedulers) > 0:
            lr_scheduler = HybridLRS(out["optimizer"], lr_schedulers) if len(lr_schedulers) > 1 else lr_schedulers[0]
            out["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return out
