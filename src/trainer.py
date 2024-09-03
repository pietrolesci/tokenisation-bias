# import shutil

# import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from huggingface_hub import HfApi
from torch import Tensor
from torch.optim import AdamW, Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.trainer import Trainer

# from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HubStrategy, IntervalStrategy
# from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME, PushInProgress
from src.optim import get_wsd_scheduler
from src.utilities import get_logger

# Configure the logger and configure colorlog
logger = get_logger("trainer", "info")
TRAINING_ARGS_NAME = "training_args.bin"

api = HfApi()


class LMTrainer(Trainer):
    def __init__(self, model, config: PretrainedConfig, data_path: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.config = config
        self.data_path = data_path

    def create_optimizer(self) -> Optimizer:
        # need to set self.optimizer

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
        self.optimizer = AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            fused=True,
        )

        return self.optimizer

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
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def get_train_dataloader(self) -> DataLoader:
        ds = DatatroveFolderDataset(
            folder_path=self.data_path,
            seq_len=self.config.max_position_embeddings,
            shuffle=True,
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

        return self.accelerator.prepare(DataLoader(ds, **dataloader_params))

    def compute_loss(self, model: LlamaForCausalLM, inputs: dict, return_outputs: bool = False) -> Tensor:
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        # return (outputs.loss, outputs) if return_outputs else outputs.loss
        return outputs.loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss

    # # def _push_from_checkpoint(self, checkpoint_folder) -> None:
    # #     Only push from one node.
    # #     if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
    # #         return

    # #     self.model.push_to_hub(repo_id=self.hub_model_id, revision=f"step{self.state.global_step}")  # type: ignore

    # #     output_dir = self.args.output_dir

    # #     # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
    # #     for modeling_file in [CONFIG_NAME, SAFE_WEIGHTS_NAME]:
    # #         if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
    # #             shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))

    # #     # Saving the tokenizer is fast and we don't know how many files it may have spawned, so we resave it to be sure.
    # #     if self.tokenizer is not None:
    # #         self.tokenizer.save_pretrained(output_dir)

    # #     # Same for the training arguments
    # #     torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # #     if self.args.save_strategy == IntervalStrategy.STEPS:
    # #         commit_message = f"Training in progress, step {self.state.global_step}"
    # #     else:
    # #         commit_message = f"Training in progress, epoch {int(self.state.epoch)}"  # type: ignore

    # #     # upload all state required to restart training to main
    # #     logger.info("Saving full state")
    # #     model_push_job = api.upload_folder(
    # #         repo_id=self.hub_model_id,  # type: ignore
    # #         folder_path=output_dir,
    # #         commit_message=commit_message,
    # #         token=self.args.hub_token,
    # #         run_as_future=True,
    # #         ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
    # #     )

    # #     push_jobs = [model_push_job]

    # #     if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
    # #         # create branch per each checkpoint and save only the model
    # #         api.create_branch(
    # #             repo_id=self.hub_model_id,  # type: ignore
    # #             branch=f"step{self.state.global_step}",
    # #             token=self.args.hub_token,
    # #             exist_ok=True,
    # #         )
    # #         logger.info("Saving checkpoint")
    # #         checkpoint_push = api.upload_folder(
    # #             repo_id=self.hub_model_id,  # type: ignore
    # #             folder_path=checkpoint_folder,
    # #             path_in_repo="./",
    # #             commit_message=f"Uploading model checkpoint to branch - step {self.state.global_step}",
    # #             token=self.args.hub_token,
    # #             run_as_future=True,
    # #             allow_patterns=[SAFE_WEIGHTS_NAME],
    # #         )
    # #         push_jobs.append(checkpoint_push)

    # #     if self.push_in_progress is None or self.push_in_progress.is_done():
    # #         self.push_in_progress = PushInProgress(push_jobs)  # type: ignore
    # #     else:
    # #         self.push_in_progress.jobs.extend(push_jobs)  # type: ignore
