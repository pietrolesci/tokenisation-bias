import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast, set_seed
from transformers.training_args import TrainingArguments

from src.model import get_model
from src.trainer import DEFAULT_TRAINING_ARGS, LMTrainer, get_scheduler_kwargs

# Configure the logger and configure colorlog
logger = logging.getLogger("hydra")


@hydra.main(version_base=None, config_path="../conf", config_name="train_conf")
def main(cfg: DictConfig) -> None:
    # Load tokenizer
    tok_path = Path(cfg.tok_path)
    logger.info(f"Loading tokenizer from {tok_path=}")
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(str(tok_path))  # type: ignore
    tok.eos_token_id = 0
    tok.pad_token_id = 0

    # Load model
    model, config = get_model(cfg.model, tok)
    logger.info(f"Model config:\n{config.to_json_string()}")
    logger.info(f"Attention implementation: {config._attn_implementation}")
    logger.info("Model", model)
    logger.info(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    logger.info(f"Num parameters: {model.num_parameters() / 1e6:.1f}M")

    # Save initial checkpoints. NOTE: manually making naming nomenclature equal to the Trainer default
    if cfg.resume_from_checkpoint is None:
        model.save_pretrained("./checkpoints/checkpoint-0")

    # Define training arguments
    training_args = TrainingArguments(
        torch_compile=cfg.torch_compile,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        num_train_epochs=cfg.num_train_epochs,
        **get_scheduler_kwargs(cfg.lr_scheduler),  # type: ignore
        **DEFAULT_TRAINING_ARGS,  # type: ignore
    )

    set_seed(training_args.seed)
    trainer = LMTrainer(
        model,
        args=training_args,
        tokenizer=tok,
        config=config,
        train_data_path=cfg.train_data_path,
        eval_data_path=cfg.eval_data_path,
        optimizer=cfg.optimizer,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # TODO: rename folder with {model_name}-{num_params}-{tok_name} automatically
    # f"{cfg.model}-{model.num_parameters() / 1e6:.0f}M-{tok_path.name}"
    # Rename current working directory to "cur_dir"
    # cur_dir = Path.cwd()
    # new_dir = cur_dir.with_name("cur_dir")
    # cur_dir.rename(new_dir)


if __name__ == "__main__":
    main()
