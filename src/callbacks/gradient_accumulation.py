from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn  # type: ignore
from typing_extensions import override

from src.utilities import get_logger

logger = get_logger("grad_accum")


class GradientAccumulationScheduler(Callback):
    def __init__(self, scheduling: dict[int, int] | None = None) -> None:
        super().__init__()
        scheduling = scheduling or {0: 1}
        minimal_step = min(scheduling.keys())
        if minimal_step < 0:
            raise IndexError(f"Step indexing from 1, step {minimal_step} cannot be interpreted correct")
        if minimal_step != 0:  # if user didn't define first step accumulation factor
            scheduling.update({0: 1})

        self.scheduling = scheduling
        self.steps = sorted(scheduling.keys())

    def going_to_accumulate_grad_batches(self) -> bool:
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, step: int) -> int:
        accumulate_grad_batches = 1
        for iter_step in reversed(self.steps):
            if step >= iter_step:
                accumulate_grad_batches = self.scheduling[iter_step]
                break
        return accumulate_grad_batches

    @override
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Performns a configuration validation before training starts and raises errors for incompatible settings."""

        if not pl_module.automatic_optimization:
            raise RuntimeError(
                """Automatic gradient accumulation and the `GradientAccumulationScheduler` is not supported for
                manual optimization. Please remove the callback or switch to automatic optimization."""
            )

        overridden_optimizer_step = is_overridden("optimizer_step", pl_module)
        overridden_optimizer_zero_grad = is_overridden("optimizer_zero_grad", pl_module)
        going_to_accumulate_grad_batches = self.going_to_accumulate_grad_batches()
        has_overridden_optimization_functions = overridden_optimizer_step or overridden_optimizer_zero_grad
        if has_overridden_optimization_functions and going_to_accumulate_grad_batches:
            rank_zero_warn(
                "When using `Trainer(accumulate_grad_batches != 1)` and overriding"
                " `LightningModule.optimizer_{step,zero_grad}`, the hooks will not be called on every batch"
                " (rather, they are called on every optimization step)."
            )

        # local import to avoid circular import
        from lightning.pytorch.strategies import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy):
            raise RuntimeError(
                f"The `{type(trainer.strategy).__name__}` does not support `accumulate_grad_batches` changing"
                " between steps."
            )
        if trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "You have set `accumulate_grad_batches` and are using the `GradientAccumulationScheduler`"
                " callback. Either remove `accumulate_grad_batches` from the Trainer or remove the callback."
            )

    @override
    def on_train_batch_start(self, trainer: Trainer, *_: Any) -> None:
        prev = trainer.accumulate_grad_batches
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(trainer.global_step)
        if trainer.accumulate_grad_batches != prev:
            logger.info(
                f"Number of accumulation batches changed from {prev} to {trainer.accumulate_grad_batches} "
                f"at step {trainer.global_step}"
            )

    # TODO: log number of elements in batch