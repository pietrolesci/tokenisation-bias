import math
from collections.abc import Callable
from typing import Literal

from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim.optimizer import Optimizer


def wsd_schedule(
    num_training_steps: int,
    num_warmup_steps: int,
    final_lr_factor: float,
    init_div_factor: float,
    frac_decay: float,
    decay_type: Literal["cosine", "mirror_cosine", "linear", "exp", "square", "sqrt"],
) -> Callable:
    # https://github.com/epfml/schedules-and-scaling/blob/6e8b7f952420c928cc09a0e4bda9678e2bf42e5f/src/optim/utils.py#L55
    num_anneal_steps = int(frac_decay * num_training_steps)
    num_hold = num_training_steps - num_anneal_steps

    assert decay_type in ["cosine", "mirror_cosine", "linear", "exp", "square", "sqrt"]

    def schedule(step) -> float:
        if step < num_warmup_steps:
            return (step / num_warmup_steps) + (1 - step / num_warmup_steps) / init_div_factor

        elif step < num_hold:
            return 1.0

        elif step < num_training_steps:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (1 - (step - num_hold) / num_anneal_steps)

            elif decay_type == "exp":
                return final_lr_factor ** ((step - num_hold) / num_anneal_steps)

            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor) * (1 + math.cos(math.pi * (step - num_hold) / num_anneal_steps)) * 0.5
                )

            elif decay_type == "mirror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor) * (1 + math.cos(math.pi * (step - num_hold) / num_anneal_steps)) * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (1 - (step - num_hold) / num_anneal_steps)
                return linear_value * 2 - cosine_value

            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (1 - ((step - num_hold) / num_anneal_steps) ** 2)

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (1 - math.sqrt((step - num_hold) / num_anneal_steps))

        else:
            return final_lr_factor

    return schedule


def get_wsd_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    final_lr_factor: float = 0.0,
    init_div_factor: float = 100,
    frac_decay: float = 0.1,
    decay_type: Literal["cosine", "mirror_cosine", "linear", "exp", "square", "sqrt"] = "linear",
) -> LRScheduler:
    lambda_schedule = wsd_schedule(
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        frac_decay=frac_decay,
        init_div_factor=init_div_factor,
        final_lr_factor=final_lr_factor,
        decay_type=decay_type,
    )
    return LambdaLR(optimizer, lambda_schedule)
