import logging
import time
from os import cpu_count
from pathlib import Path

import hydra
import polars as pl
import srsly
import torch
from datasets import load_from_disk
from lightning.fabric import Fabric, seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_warning
import numpy as np
SEP_LINE = f"{'=' * 70}"
MODEL_CACHE_DIR = ".model_cache"

log = logging.getLogger("hydra")

set_verbosity_warning()


def flatten(x: list[list]) -> list:
    return [i for j in x for i in j]


def jsonl2parquet(filepath: str | Path, out_dir: str | Path) -> None:
    filepath = Path(filepath)
    assert filepath.name.endswith(".jsonl"), "Not a jsonl file"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl = srsly.read_jsonl(filepath)
    df = pl.DataFrame({k: flatten(v) for k, v in ld_to_dl(line).items()} for line in fl)  # type: ignore
    df = df.explode(df.columns)

    df.write_parquet(out_dir / f"{filepath.name.removesuffix('.jsonl')}.parquet")


def ld_to_dl(ld: list[dict]) -> dict[str, list]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def collator_fn(batch: list[dict[str, int | list[int]]]) -> dict:
    new_batch = ld_to_dl(batch)

    # Pad from the left side
    input_ids: list[list[int]] = new_batch["input_ids"]
    max_length = max(len(x) for x in input_ids)
    padded_input_ids = np.vstack([
        # to understand this function: pad_before == max_length - len(x) and pad_after == 0
        np.pad(x, (max_length - len(x), 0), mode='constant', constant_values=0)
        for x in input_ids
    ])
    new_batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)  # type: ignore

    return new_batch


def compute_statistics(model: torch.nn.Module, batch: dict) -> dict:
    """More efficient alternative that manually computes surprisal"""
    # Forward pass
    input_ids = batch["input_ids"]
    labels = input_ids.clone()
    
    # Shift so that tokens < n predict n
    # shift_input_ids = input_ids[..., :-1].contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    # shift_logprobs = model.forward(input_ids=shift_input_ids).logits.log_softmax(-1)

    logprobs = model.forward(input_ids=input_ids).logits.log_softmax(-1)
    shift_logprobs = logprobs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Get the log-probability of the true token
    # (batch, seq, vocab = 1), there is an extra dim that makes it broadcastable to `shift_logprobs`
    true_logprobs = shift_logprobs.take_along_dim(dim=-1, indices=shift_labels[..., None])

    # Get surprisal, aka the negative of the log-probability
    # this returns equal results to torch.nn.functional.cross_entropy (atol: 1e-3)
    # `torch.nn.functional.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none")`
    # but we gain the fact that we can compute other statistics of the true_logprobs other that the surprisal
    sup = true_logprobs.squeeze(-1).neg()

    # Get the rank of the true token
    # how many bigger token have log-probability bigger than the true token? this is the rank of the true token
    rank = (shift_logprobs > true_logprobs).long().sum(-1)

    # Get the entropy -\sum logp * p
    entropy = (shift_logprobs * shift_logprobs.exp()).sum(-1).neg()

    # Keep only the last two tokens
    return {
        "new_token_id": batch["new_token_id"],
        "uid": batch["uid"],
        "input_ids": input_ids[:, -2:].cpu().numpy().tolist(),
        "sup": sup[:, -2:].cpu().numpy().tolist(),  # convertion to numpy works because model is outputting float32 somehow
        "rank": rank[:, -2:].cpu().numpy().tolist(),
        "entropy": entropy[:, -2:].cpu().numpy().tolist(),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="eval_conf")
def main(cfg: DictConfig) -> None:
    # =============================
    # Step 1. Prepare configuration
    # =============================
    start_time = time.time()
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    seed_everything(cfg.seed)
    log.info(f"Seed enabled: {cfg.seed}")

    # ===========================
    # Step 2. Load model and data
    # ===========================
    model_path = Path(cfg.model_path)     
    model = AutoModelForCausalLM.from_pretrained(str(model_path / "checkpoints" / cfg.checkpoint))
    
    tok_name: str = srsly.read_yaml(model_path / "metadata.yaml")["tok_name"]  # type: ignore
    data_path = Path(cfg.data_path + f"-{tok_name}")
    dataset = load_from_disk(str(data_path / "contexts"))

    data_loader = DataLoader(
        dataset,  # type: ignore
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cpu_count(),  # type: ignore
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collator_fn,
        multiprocessing_context=cfg.multiprocessing_context,
    )

    # =====================
    # Step 4. Run inference
    # =====================
    torch.set_float32_matmul_precision(cfg.tf32_mode)

    fabric = Fabric(accelerator=cfg.accelerator, precision=cfg.precision)
    model = fabric.setup_module(model)
    model.eval()

    if cfg.torch_compile:
        model = torch.compile(model)

    data_loader = fabric.setup_dataloaders(data_loader, use_distributed_sampler=False)

    FILENAME = model_path.name
    write_buffer = []
    pbar = tqdm(data_loader, desc="Running Inference")
    for idx, batch in enumerate(pbar):
        pbar.set_postfix_str(f"Buffer size: {len(write_buffer)}")

        # show a batch
        if idx == 0:
            log.info(f"A batch: {batch}")

        with torch.inference_mode():
            out = compute_statistics(model, batch)  # type: ignore
        write_buffer.append(out)

        # Write to disk
        if len(write_buffer) == cfg.write_interval or idx == len(pbar):
            pbar.set_description_str("Writing to disk")
            srsly.write_jsonl(f"{FILENAME}.jsonl", [write_buffer], append=True)

            pbar.set_description_str("Running Inference")
            write_buffer = []

    # clean-up
    jsonl2parquet(filepath=f"{FILENAME}.jsonl", out_dir=".")
    log.info(f"Total time: {(time.time() - start_time) // 60} minutes")


if __name__ == "__main__":
    main()
