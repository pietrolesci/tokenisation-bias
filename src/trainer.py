# import shutil

# import torch
import os
from typing import Literal

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from datatrove.utils.dataset import DatatroveFolderDataset
from huggingface_hub import HfApi
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.trainer import Trainer

# from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HubStrategy, IntervalStrategy
# from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME, PushInProgress
from src.optim import ZeroShampooWithAdamGraftingOptimizer, get_wsd_scheduler
from src.utilities import get_logger, ld_to_dl

# Configure the logger and configure colorlog
logger = get_logger("trainer", "info")
TRAINING_ARGS_NAME = "training_args.bin"

api = HfApi()


class LMTrainer(Trainer):
    def __init__(
        self,
        model,
        config: PretrainedConfig,
        train_data_path: str,
        eval_data_path: str | None = None,
        optimizer: Literal["adamw", "shampoo"] = "adamw",
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        self.config = config
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.optimizer_class = optimizer

    def create_optimizer(self) -> Optimizer:
        # Get params that require grad
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version

        if self.optimizer_class == "adamw":
            optim = AdamW(
                optim_groups,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                fused=True,
            )
        else:
            optim = ZeroShampooWithAdamGraftingOptimizer(
                optim_groups, lr=self.args.learning_rate, device=self.accelerator.device
            )

        self.optimizer = optim
        logger.info(f"{self.optimizer=}")

        return self.optimizer  # type: ignore

    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer) -> LRScheduler:
        # HACK: to avoid changing too much stuff, just assume that when I pass kwargs
        # I mean that I want the wsd scheduler
        if isinstance(self.args.lr_scheduler_kwargs, dict):
            self.lr_scheduler = get_wsd_scheduler(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
        else:
            self.lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        logger.info(f"{self.lr_scheduler}")
        return self.lr_scheduler

    def get_train_dataloader(self) -> DataLoader:
        ds = DatatroveFolderDataset(
            folder_path=self.train_data_path,
            filename_pattern=f"{self.train_data_path}/*.ds",
            seq_len=self.config.max_position_embeddings,
            shuffle=False,
            seed=42,
            token_size=2 if self.config.vocab_size < 65_000 else 4,
        )
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": True,
            "persistent_workers": False,
            "shuffle": False,
            "drop_last": self.args.dataloader_drop_last,
        }
        logger.info(f"{ds=}")
        logger.info(f"{dataloader_params=}")
        return self.accelerator.prepare(DataLoader(ds, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: str | TorchDataset | None = None) -> DataLoader:
        if self.eval_data_path is None:
            return super().get_eval_dataloader(eval_dataset)

        max_length = self.model.config.max_position_embeddings

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

        ds: Dataset = load_from_disk(self.eval_data_path)  # type: ignore
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": collator_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": True,
            "persistent_workers": False,
            "shuffle": False,
            "drop_last": self.args.dataloader_drop_last,
        }
        logger.info(f"{ds=}")
        logger.info(f"{dataloader_params=}")
        return self.accelerator.prepare(DataLoader(ds, **dataloader_params))  # type: ignore

    def compute_loss(self, model: LlamaForCausalLM, inputs: dict, return_outputs: bool = False) -> Tensor:
        # This is the typical way, but it is unclear to me why I need to pass seq_len+1
        # and internally transformers does not seem to complain, but runs inference on seq_len+1 tokens.
        # However, I do not need the predicted next token at position seq_len+1
        # >>> input_ids = inputs["input_ids"]
        # >>> labels = input_ids.clone()
        # >>> outputs = model(input_ids=input_ids, labels=labels)
        input_ids = inputs["input_ids"][:, :-1]
        labels = inputs["input_ids"][:, 1:]

        logits = model.forward(input_ids=input_ids).logits  # type: ignore
        loss = cross_entropy(logits.permute(0, 2, 1), labels)
        return loss

    def prediction_step(
        self,
        model: LlamaForCausalLM,
        inputs: dict[str, Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple:
        with torch.inference_mode(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs).detach()
        return (loss, None, None)


DEFAULT_TRAINING_ARGS = dict(
    # =======
    # logging
    # =======
    output_dir="./checkpoints",  # hydra takes care of moving into correct experiment folder
    logging_dir="./",  # hydra takes care of moving into correct experiment folder
    logging_strategy="steps",
    logging_first_step=True,
    log_level="passive",  # takes it from global
    logging_steps=1,
    report_to="tensorboard",
    include_num_input_tokens_seen=True,
    # =============
    # checkpointing
    # =============
    eval_on_start=True,
    eval_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    save_safetensors=True,
    # ===========
    # push to hub
    # ===========
    # push_to_hub=True,
    # hub_model_id=model_name,
    # hub_strategy="all_checkpoints",
    # hub_private_repo=True,
    # =====
    # setup
    # =====
    seed=42,
    bf16=True,
    bf16_full_eval=True,
    tf32=True,
    # ============
    # optimisation
    # ============
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    # ===========
    # dataloading
    # ===========
    dataloader_num_workers=os.cpu_count() - 1,  # type: ignore
    dataloader_pin_memory=True,
)


def get_scheduler_kwargs(name: str) -> dict:
    if name == "wsd":
        return {
            "lr_scheduler_kwargs": {
                "final_lr_factor": 0.0,
                "init_div_factor": 100,
                "frac_decay": 0.2,
                "decay_type": "sqrt",
            }
        }
    return {"lr_scheduler_type": name, "lr_scheduler_kwargs": None}
