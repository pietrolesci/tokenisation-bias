import logging
import time
from collections.abc import Generator
from pathlib import Path

import hydra
import numpy as np
import srsly
import torch
from datasets import Dataset, load_from_disk
from lightning.fabric import Fabric, seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm

from primer.trainer import load_hf_from_pl
from primer.utilities import jsonl2parquet, ld_to_dl, remove_file

SEP_LINE = f"{'=' * 80}"
OmegaConf.register_new_resolver(
    "get_folder_name", lambda x, y: Path(x).name if x is not None else Path(y.split("/")[1]).name
)

# Configure the logger and configure colorlog
logger = logging.getLogger("hydra")


def batch_by_tokens_generator_with_padding(
    sorted_dataset: Dataset, max_tokens_per_batch: int
) -> Generator[list, None, None]:
    """
    A generator that yields batches of documents such that the total number of tokens
    in each batch does not exceed `max_tokens_per_batch`, considering padding.
    """
    current_batch = []
    current_max_length = 0

    for example in sorted_dataset:
        current_batch.append(example)

        num_tokens = len(example["input_ids"])  # type: ignore

        # Determine the new maximum length if we add this document
        new_max_length = max(current_max_length, num_tokens)

        # Calculate the total tokens with padding if this document is added
        new_total_tokens = new_max_length * (len(current_batch) + 1)

        if new_total_tokens > max_tokens_per_batch:
            # Yield the current batch and start a new one
            yield current_batch
            current_batch = []
            current_max_length = 0
        else:
            # Add the document to the current batch
            current_max_length = new_max_length

    # Yield the final batch
    if current_batch:
        yield current_batch


def collator_fn(batch: list[dict[str, int | list[int]]], pad_value: int = 0) -> dict:
    new_batch = ld_to_dl(batch)

    # Pad from the left side
    input_ids: list[list[int]] = new_batch["input_ids"]
    max_length = max(len(x) for x in input_ids)
    padded_input_ids = np.vstack(
        [
            # to understand this function: pad_before == max_length - len(x) and pad_after == 0
            np.pad(x, (max_length - len(x), 0), mode="constant", constant_values=pad_value)
            for x in input_ids
        ]
    )
    new_batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)  # type: ignore

    return new_batch


def compute_statistics_all_tokens(
    model: torch.nn.Module, batch: dict, pad_value: int = 0, window_size: int | None = 512, step_size: int | None = 256
) -> dict:
    """Efficient alternative that computes surprisal using a sliding window approach."""

    # Initialise outputs with metadata
    out = {"uid": batch["uid"]}
    input_ids = batch["input_ids"]
    seq_len = input_ids.shape[1]

    # When we pass None, process the entire sequence in one go
    if window_size is None:
        window_size: int = seq_len
        step_size: int = seq_len  # makes sure there's only one iteration

    assert window_size >= step_size  # type: ignore

    all_logprobs = []
    token_ids = []
    prev_end = 0
    with torch.inference_mode():
        for start in range(0, seq_len - 1, step_size):
            end = min(start + window_size, seq_len - 1)

            # Extract windowed input and labels (shifted left for next-token prediction)
            input_chunk = input_ids[:, start:end]
            label_chunk = input_ids[:, start + 1 : end + 1]

            # Compute logits
            logits = model(input_chunk).logits  # Shape: (batch_size, chunk_len, vocab_size)
            logprob_chunk = cross_entropy(
                logits.permute(0, 2, 1), label_chunk, reduction="none", ignore_index=pad_value
            ).neg()

            if start != 0:
                s = min(end - prev_end, step_size)
                logprob_chunk = logprob_chunk[:, -s:]
                label_chunk = label_chunk[:, -s:]

            all_logprobs.append(logprob_chunk.cpu())
            token_ids.append(label_chunk.cpu())

            prev_end = end
            if end == seq_len - 1:
                break

    # Concatenate results along sequence length
    out["token_logprob"] = torch.cat(all_logprobs, dim=1).numpy().tolist()
    out["token_ids"] = torch.cat(token_ids, dim=1).numpy().tolist()

    assert seq_len - 1 == len(out["token_ids"][0]), f"{seq_len - 1} {len(out['token_ids'][0])}"

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
    if cfg.run_path:
        # Get metadata
        run_path = Path(cfg.run_path)
        tok_path = Path(srsly.read_yaml(run_path / "hparams.yaml")["tok_path"])  # type: ignore

        # Load model
        ckpt_path = run_path / ".checkpoints" / f"{cfg.checkpoint}.ckpt"
        model = load_hf_from_pl(ckpt_path)

    elif all([cfg.tok_path, cfg.repo_id, cfg.checkpoint]):
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading model from {cfg.repo_id=}, {cfg.checkpoint=}")
        model = AutoModelForCausalLM.from_pretrained(cfg.repo_id, revision=cfg.checkpoint, cache_dir=".model_cache")
        tok_path = Path(cfg.tok_path)

    else:
        raise ValueError("Either `run_path` or `tok_path`, `repo_id`, and `revision` must be provided")

    # Load data
    data_path = Path(cfg.data_path) / tok_path.name / cfg.data_split
    logger.info(f"Loading data from {data_path=}")
    dataset: Dataset = load_from_disk(data_path)  # type: ignore
    dataset = (
        dataset.map(
            lambda x: {"length": [len(s) for s in x["input_ids"]]},
            load_from_cache_file=False,
            keep_in_memory=True,
            batched=True,
            num_proc=8,
        )
        # longer to shorter
        .sort("length", reverse=True, load_from_cache_file=False, keep_in_memory=True)
    )
    logger.info(f"Loaded {len(dataset)} documents")
    logger.info(f"Loaded {dataset[:10]['length']}")
    batch_generator = batch_by_tokens_generator_with_padding(dataset, cfg.max_tokens_per_batch)

    # =====================
    # Step 4. Run inference
    # =====================
    torch.set_float32_matmul_precision("high")

    # Set model with Fabric
    fabric = Fabric(accelerator=cfg.accelerator, precision=cfg.precision)
    model = fabric.setup_module(model)

    # Compile
    if cfg.torch_compile:
        model.compile()

    # Eval mode
    model.eval()

    FILENAME = cfg.checkpoint
    write_buffer = []
    pbar = tqdm(total=len(dataset), desc="Running Inference")
    num_docs = 0
    for batch in batch_generator:
        # Increment doc counter
        current_num_docs = len(batch)
        num_docs += current_num_docs

        # Construct batch
        batch = collator_fn(batch)
        pbar.set_postfix_str(f"Buffer: {len(write_buffer)}, Batch: {batch['input_ids'].shape}")
        batch = fabric.to_device(batch)

        # Compute statistics
        with torch.inference_mode():
            out = compute_statistics_all_tokens(
                model,
                batch,
                pad_value=0,
                # If window_size is larger than the sequence length, process the entire sequence in one go
                window_size=cfg.window_size if batch["input_ids"].shape[1] > cfg.window_size else None,
                step_size=cfg.step_size,
            )
        write_buffer.append(out)

        # Write to disk
        if len(write_buffer) == cfg.write_interval or num_docs == pbar.total - 1:
            pbar.set_description_str("Writing to disk")
            srsly.write_jsonl(f"{FILENAME}.jsonl", [write_buffer], append=True)
            pbar.set_description_str("Running Inference")
            write_buffer = []

        pbar.update(current_num_docs)

    # Write any remaining data in the buffer after the loop
    if write_buffer:
        pbar.set_description_str("Writing remaining data to disk")
        srsly.write_jsonl(f"{FILENAME}.jsonl", [write_buffer], append=True)
        pbar.set_description_str("Inference complete")

    # Clean up
    jsonl2parquet(filepath=f"{FILENAME}.jsonl", out_dir=".")
    if Path(f"{FILENAME}.parquet").exists():
        remove_file(f"{FILENAME}.jsonl")
    logger.info(f"Total time: {(time.time() - start_time) // 60} minutes")


if __name__ == "__main__":
    main()
