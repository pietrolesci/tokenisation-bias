import logging
from pathlib import Path

import hydra
import srsly
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast, set_seed
from transformers.training_args import TrainingArguments

from src.model import get_model
from src.trainer import DEFAULT_TRAINING_ARGS, LMTrainer, get_scheduler_kwargs

# Configure the logger and configure colorlog
logger = logging.getLogger("hydra")


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
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
    model.save_pretrained("./checkpoints/checkpoint-0")
    srsly.write_yaml(
        "model_name.yaml",
        {"model_name": f"{cfg.model}-{model.num_parameters() / 1e6:.0f}M-{tok_path.name}"},
    )

    # Define training arguments
    training_args = TrainingArguments(
        torch_compile=cfg.torch_compile,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        **get_scheduler_kwargs(cfg.lr_scheduler),  # type: ignore
        **DEFAULT_TRAINING_ARGS,  # type: ignore
    )

    set_seed(training_args.seed)
    trainer = LMTrainer(
        model, args=training_args, tokenizer=tok, config=config, data_path=f"{cfg.dataset_repo}/{tok_path.name}"
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)


if __name__ == "__main__":
    main()
