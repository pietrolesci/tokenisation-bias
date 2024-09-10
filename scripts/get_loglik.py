import logging
import time
from os import cpu_count
from pathlib import Path

import hydra
import srsly
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from lightning.fabric import seed_everything, Fabric
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.utils.logging import set_verbosity_warning
from transformers import AutoModelForCausalLM
import torch
import polars as pl

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

def collator_fn(batch: list[dict[str, int | list[int]]]) -> dict[str, list[int] | torch.LongTensor]:
    new_batch = ld_to_dl(batch)
    new_batch["contexts"] = torch.tensor(new_batch["contexts"], dtype=torch.long)  # type: ignore
    return new_batch  # type: ignore

def compute_statistics(model: torch.nn.Module, token_ids: torch.LongTensor) -> dict:
        """More efficient alternative that manually computes surprisal"""
        # Forward pass
        labels = token_ids.clone()
        logprobs = model.forward(token_ids).log_softmax(-1)

        # Shift so that tokens < n predict n
        shift_labels = labels[..., 1:].contiguous()
        shift_logprobs = logprobs[..., :-1, :].contiguous()

        # Get the log-probability of the true token
        # (batch, seq, vocab = 1), there is an extra dim that makes it broadcastable to `shift_logprobs`
        true_logprobs = shift_logprobs.take_along_dim(dim=-1, indices=shift_labels[..., None])

        # Get surprisal, aka the negative of the log-probability
        sup = true_logprobs.squeeze(-1).neg()

        # Get the rank of the true token
        # how many bigger token have log-probability bigger than the true token? this is the rank of the true token
        rank = (shift_logprobs > true_logprobs).long().sum(-1)

        # Get the entropy
        #  - \sum logp * p
        entropy = (shift_logprobs * shift_logprobs.exp()).sum(-1).neg()

        return {
            "sup": sup.cpu().numpy().tolist(),  # convertion to numpy works because model is outputting float32 somehow
            "rank": rank.cpu().numpy().tolist(),
            "entropy": entropy.cpu().numpy().tolist(),
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
    tok_name: str = srsly.read_yaml(model_path / "metadata.yaml")["tok_name"]  # type: ignore
    data_path = Path(cfg.data_path + f"-{tok_name}")
    assert (model_path / "checkpoints" / cfg.checkpoint).exists(), f"{model_path=} and {cfg.checkpoint=} does not exist!"
    assert data_path.exists(), f"{data_path=} does not exist!"    

    model = AutoModelForCausalLM.from_pretrained(str(model_path / "checkpoints" / cfg.checkpoint))
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
    
    log.info("Running inference")
    pbar = tqdm(data_loader, desc="Running Inference")
    write_buffer = []
    FILENAME = model_path.name

    for idx, batch in enumerate(pbar):
        pbar.set_postfix_str(f"Buffer size: {len(write_buffer)}")

        with torch.inference_mode():
            out = compute_statistics(model, batch["contexts"])  # type: ignore
        out["seq_idx"] = batch["seq_idx"]
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