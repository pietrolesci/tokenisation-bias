import logging
import time
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.datamodule import DataloaderConfig, DataModule
from src.model import get_model
from src.module import LanguageModel, OptimCofig
from src.utilities import conf_to_dict, instantiate_from_conf

SEP_LINE = f"{'=' * 80}"

# Configure the logger and configure colorlog
logger = logging.getLogger("hydra")


@hydra.main(version_base=None, config_path="../conf", config_name="train_conf_pl")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    # Load tokenizer
    tok_path = Path(cfg.tok_path)
    logger.info(f"Loading tokenizer from {tok_path=}")
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(str(tok_path))  # type: ignore
    tok.eos_token_id = 0
    tok.pad_token_id = 0

    # Load model
    model, _ = get_model(cfg.model, tok)
    logger.info(f"Model config:\n{model.config.to_json_string()}")
    logger.info(f"Attention implementation: {model.config._attn_implementation}")
    logger.info(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    logger.info(f"Num parameters: {model.num_parameters() / 1e6:.1f}M")

    # Save initial checkpoints. NOTE: manually making naming nomenclature equal to the Trainer default
    # if cfg.resume_from_checkpoint is None:
    #     model.save_pretrained("./checkpoints/checkpoint-0")

    dataloader_config = DataloaderConfig(**conf_to_dict(cfg.data))  # type: ignore
    datamodule = DataModule(
        train_data_path=cfg.train_data_path,
        val_data_path=cfg.val_data_path,
        max_position_embeddings=model.config.max_position_embeddings,
        dataloader_config=dataloader_config,
    )
    optim_config = OptimCofig(**conf_to_dict(cfg.optim))  # type: ignore

    if cfg.torch_compile:
        model = torch.compile(model)
    module = LanguageModel(model, optim_config)  # type: ignore
    loggers, callbacks = instantiate_from_conf([cfg.get(i) for i in ("loggers", "callbacks")])

    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    trainer = Trainer(**conf_to_dict(cfg.trainer), logger=loggers, callbacks=callbacks)
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=cfg.resume_from_checkpoint)
    logger.info(f"Training total time: {(time.perf_counter() - start_time) / 60:.1f} minutes")

    # TODO: rename folder with {model_name}-{num_params}-{tok_name} automatically
    # f"{cfg.model}-{model.num_parameters() / 1e6:.0f}M-{tok_path.name}"
    # Rename current working directory to "cur_dir"
    # cur_dir = Path.cwd()
    # new_dir = cur_dir.with_name("cur_dir")
    # cur_dir.rename(new_dir)


if __name__ == "__main__":
    main()
