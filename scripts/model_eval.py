import logging
import time
from collections.abc import Callable
from os import cpu_count
from pathlib import Path

import hydra
import numpy as np
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

SEP_LINE = f"{'=' * 80}"
MODEL_CACHE_DIR = ".model_cache"

logger = logging.getLogger("hydra")

set_verbosity_warning()

OmegaConf.register_new_resolver("get_folder_name", lambda x: Path(x).name)
OmegaConf.register_new_resolver("get_tok_name", lambda x: Path(x).name.split("-")[-1])


def flatten(x: list[list]) -> list:
    return [i for j in x for i in j]


def remove_file(path: str | Path) -> None:
    path = Path(path)
    path.unlink(missing_ok=True)


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


def get_collator_fn(prefix_map: dict) -> Callable:
    def collator_fn(batch: list[dict[str, int | list[int]]]) -> dict:
        new_batch = ld_to_dl(batch)

        # Pad from the left side
        input_ids: list[list[int]] = new_batch["input_ids"]
        max_length = max(len(x) for x in input_ids)
        padded_input_ids = np.vstack(
            [
                # to understand this function: pad_before == max_length - len(x) and pad_after == 0
                np.pad(x, (max_length - len(x), 0), mode="constant", constant_values=0)
                for x in input_ids
            ]
        )
        new_batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)  # type: ignore

        # Create a list[list[int]] with the locations, for each token. of where in the vocab it needs to sum
        # new_batch["vocab_locations"] = [
        #     [i] + prefix_map.get(i, []) for i in new_batch["input_ids"][:, -2:].flatten().tolist()
        # ]
        vocab_locations: list[tuple[int, list[int]]] = []
        for penultimate_token, last_token in new_batch["input_ids"][:, -2:].unbind():
            # Penultimate token only needs itself (always)
            m_penultimate: int = penultimate_token.item()  # type: ignore
            # For the last token we apply the trick
            m_last = [last_token.item()] + prefix_map.get(last_token.item(), [])
            vocab_locations.append((m_penultimate, m_last))
        new_batch["vocab_locations"] = vocab_locations

        return new_batch

    return collator_fn


def load_prefix_map(path: str | Path) -> dict[int, list[int]]:
    prefix_map: dict[int, list[int]] = {d["prefix"]: d["new_token_id"] for d in srsly.read_jsonl(path)}  # type: ignore
    logger.info(f"Check that prefix map is only available for in-vocab tokens. Last tok is: {max(prefix_map.keys())}")
    return prefix_map


def compute_statistics(model: torch.nn.Module, batch: dict) -> dict:
    """More efficient alternative that manually computes surprisal"""

    # Initialise outputs with metadata
    out = {"new_token_id": batch["new_token_id"], "uid": batch["uid"]}

    # Forward pass and compute probs
    input_ids = batch["input_ids"]
    logits = model.forward(input_ids=input_ids[:, :-1]).logits
    probs = logits.softmax(-1)

    # Consider only the probs of the last two tokens
    last_tokens_probs = probs[:, -2:, :]

    # For the prob of the true token, instead of creating map, simply take it
    # along the dimension. Leaving implementation here for convenience
    # >>> B, S, V = last_tokens_probs.shape
    # >>> true_token_mapping = input_ids[:, -2:].flatten().tolist()
    # >>> mask_true = torch.zeros((B * S, V))
    # >>> for idx, locations in enumerate(true_token_mapping):
    # >>>     mask_true[idx, locations] = 1
    # >>> mask_true = mask_true.reshape(B, S, V)
    # >>> true_token_prob = (last_tokens_probs * mask_true).sum(-1)
    token_prob = last_tokens_probs.take_along_dim(dim=-1, indices=input_ids[:, -2:][..., None]).squeeze(-1)

    # For the probs of the true token summed with the probs of the other tokens having
    # that token as a prefix, we need to create the mask
    # vocab_locations = batch["vocab_locations"]
    # B, S, V = last_tokens_probs.shape
    # mask = torch.zeros((B * S, V), device=last_tokens_probs.device, dtype=last_tokens_probs.dtype)
    # for idx, locations in enumerate(vocab_locations):
    #     mask[idx, locations] = 1.
    # mask = mask.reshape(B, S, V)
    mask = torch.zeros_like(last_tokens_probs)
    for idx, (m_penultimate, m_last) in enumerate(batch["vocab_locations"]):
        # Penultimate token only gets its position
        mask[idx, -2, m_penultimate] = 1.0
        # Last token gets the fix
        mask[idx, -1, m_last] = 1.0

    # Then compute it as the sum of the prob assigned to the token plus the prob assigned
    # to all the other tokens which have that token as a prefix
    token_prob_fix = (last_tokens_probs * mask).sum(-1)

    # Add results to output
    out["prob_true"] = token_prob.cpu().numpy().tolist()
    out["prob_true_and_prefix"] = token_prob_fix.cpu().numpy().tolist()

    return out


@hydra.main(version_base=None, config_path="../conf", config_name="eval_conf")
def main(cfg: DictConfig) -> None:
    # =============================
    # Step 1. Prepare configuration
    # =============================
    start_time = time.time()
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    seed_everything(cfg.seed)
    logger.info(f"Seed enabled: {cfg.seed}")

    # ===========================
    # Step 2. Load model and data
    # ===========================
    model_path = Path(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(str(model_path / "checkpoints" / cfg.checkpoint))

    data_path = Path(cfg.data_path)
    dataset = load_from_disk(data_path / "eval_samples")

    prefix_map = load_prefix_map(cfg.prefix_map_path)

    dataloader = DataLoader(
        dataset,  # type: ignore
        batch_size=cfg.batch_size,
        num_workers=min(cpu_count(), 32),  # type: ignore
        collate_fn=get_collator_fn(prefix_map),
        multiprocessing_context=cfg.multiprocessing_context,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
    )

    # =====================
    # Step 4. Run inference
    # =====================
    torch.set_float32_matmul_precision(cfg.tf32_mode)

    fabric = Fabric(accelerator=cfg.accelerator, precision=cfg.precision)
    dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)
    model = fabric.setup_module(model)

    model.eval()
    if cfg.torch_compile:
        model = torch.compile(model)

    FILENAME = f"{model_path.name}_{cfg.checkpoint}"
    write_buffer = []
    pbar = tqdm(dataloader, desc="Running Inference")
    for idx, batch in enumerate(pbar):
        pbar.set_postfix_str(f"Buffer size: {len(write_buffer)}")

        # # Show a batch
        # if idx == 0:
        #     logger.info(f"A batch: {batch['input_ids']}")

        # Compute statistics
        with torch.inference_mode():
            out = compute_statistics(model, batch)  # type: ignore
        write_buffer.append(out)

        # Write to disk
        if len(write_buffer) == cfg.write_interval or idx == len(pbar) - 1:
            pbar.set_description_str("Writing to disk")
            srsly.write_jsonl(f"{FILENAME}.jsonl", [write_buffer], append=True)

            pbar.set_description_str("Running Inference")
            write_buffer = []

    # clean-up
    jsonl2parquet(filepath=f"{FILENAME}.jsonl", out_dir=".")
    if Path(f"{FILENAME}.parquet").exists():
        remove_file(f"{FILENAME}.jsonl")
    logger.info(f"Total time: {(time.time() - start_time) // 60} minutes")


if __name__ == "__main__":
    main()
