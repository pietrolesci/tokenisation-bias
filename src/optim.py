import math
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim.optimizer import Optimizer


class HybridOptim(Optimizer):
    """
    Wrapper around multiple optimizers that should be stepped together at a single time. This is
    a hack to avoid PyTorch Lightning calling ``training_step`` once for each optimizer, which
    increases training time and is not always necessary.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687

    Parameters
    ----------
    optimizers: list of optimizers

    """

    def __init__(self, optimizers: list[Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def state(self) -> dict:
        """Return the combined state for each optimizer in ``self.optimizers``."""
        return {key: value for optimizer in self.optimizers for key, value in optimizer.state.items()}

    @property
    def param_groups(self) -> list[dict[str, torch.Tensor | float | bool | Any]]:
        """Return the combined parameter groups for each optimizer in ``self.optimizers``."""
        return [element for optimizer in self.optimizers for element in optimizer.param_groups]

    # !!!!!! HERE IS THE NEW CODE !!!!!!!!!
    @property
    def defaults(self) -> dict[str, torch.Tensor]:
        """Return the combined defaults for each optimizer in ``self.optimizers``."""
        return {key: value for optimizer in self.optimizers for key, value in optimizer.defaults.items()}

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def __getstate__(self) -> list[Optimizer]:
        """Return ``self.optimizers`` for pickling purposes."""
        return self.optimizers

    def __setstate__(self, optimizers: list[Optimizer]) -> None:
        """
        Load ``optimizers`` into ``self.optimizers`` for pickling purposes and call
        ``__setstate__``.

        """
        self.optimizers = optimizers

        # call remaining lines of the ``Optimizer.__setstate__`` method just to be safe.
        # copied from: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()  # To support multiprocessing pickle/unpickle.
            optimizer.defaults.setdefault("differentiable", False)

    def __repr__(self) -> str:
        """Call and concatenate ``__repr__`` for each optimizer in ``self.optimizers``."""
        repr_str = f"``{self.__class__.__name__}`` containing {len(self.optimizers)} optimizers:\n"

        for optimizer in self.optimizers:
            repr_str += "\n" + optimizer.__repr__()

        return repr_str

    def _hook_for_profile(self) -> None:
        """Call ``_hook_for_profile`` for each optimizer in ``self.optimizers``."""
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()

    def state_dict(self) -> list[dict[str, torch.Tensor | list[dict[str, torch.Tensor | float | bool | Any]]]]:
        """
        Returns the state of the optimizer as a dictionary.

        It contains two entries:

            * ``state`` - a dict holding current optimization state. Its content differs between
              optimizer classes.
            * ``param_groups`` - a list containing all parameter groups where each parameter group
              is a dict

        """
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(
        self, state_dict: list[dict[str, torch.Tensor | list[dict[str, torch.Tensor | float | bool | Any]]]]
    ) -> None:
        """
        Loads the optimizer state.

        Parameters
        ----------
        state_dict: dict
            Optimizer state. Should be an object returned from a call to ``state_dict()``

        """
        for state, optimizer in zip(state_dict, self.optimizers, strict=True):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Sets the gradients of all optimized ``torch.Tensor``s to zero.

        Parameters
        ----------
        set_to_none: bool
            Instead of setting to zero, set the grads to ``None``. This will in general have lower
            memory footprint, and can modestly improve performance. However, it changes certain
            behaviors. For example:

                1. When the user tries to access a gradient and perform manual ops on it, a ``None``
                   attribute or a ``torch.Tensor`` full of ``0``s will behave differently.

                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass,
                   ``.grad``s are guaranteed to be ``None`` for params that did not receive a
                   gradient.

                3. ``torch.optim`` optimizers have a different behavior if the gradient is ``0`` or
                   ``None`` (in one case it does the step with a gradient of ``0`` and in the other
                   it skips the step altogether).

        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """
        Performs a single optimization step (parameter update).

        Parameters
        ----------
        closure: function
            A closure that reevaluates the model and returns the loss. Optional for most optimizers.

        Notes
        -----
        Unless otherwise specified, this function should not modify the ``.grad`` field of the
        parameters.

        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss


class HybridLRS(LRScheduler):
    """
    Wrapper class around ``lr_scheduler``s to return a dummy optimizer to pass PyTorch Lightning
    checks.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687

    Parameters
    ----------
    hybrid_optimizer: HybridOptim
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    idx: int
        Index of the optimizer in ``hybrid_optimizer`` the learning rate scheduler ``lr_scheduler``
        is assigned to

    """

    def __init__(self, optimizer: HybridOptim, lr_schedulers: list[torch.optim.lr_scheduler._LRScheduler]) -> None:
        self.lr_schedulers = lr_schedulers
        self.optimizer = optimizer  # HACK to make lightning happy during checks

    def step(self, epoch: int | None = None) -> None:
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step(epoch)

class OrthogonalNesterov(Optimizer):
    """
    Some warnings: This optimizer assumes that all parameters passed in are 2D.
    It shouldn't be used for the embedding layer, the final fully connected layer, or {0,1}-D
    parameters; those should be optimized by a standard method (e.g., AdamW).
    To use it with 4D convolutional filters, it works well to flatten their last 3 dimensions.
    """

    def __init__(
        self, params, lr: float = 0.02, momentum: float = 0.9, nesterov: bool = True, zeropower_iters: int = 5
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, zeropower_iters=zeropower_iters)
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeroth_power_via_newtonschulz5(g, steps=group["zeropower_iters"])
                scale = update.numel() ** 0.5 / update.norm()
                p.data.add_(update, alpha=-lr * scale)


@torch.compile
def zeroth_power_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> Any:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


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
