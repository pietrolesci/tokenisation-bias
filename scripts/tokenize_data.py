import os
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_from_disk
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.utils.dataset import DatatroveFolderDataset
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,  # type: ignore
)

from src.utilities import get_logger, load_tokenizer_with_vocab_size

# Configure the logger and configure colorlog
logger = get_logger("tok-inference", "info")


# Global options
HF_URL = "hf://datasets"
USERNAME = "pietrolesci"
LIMIT = -1  # NOTE: set to -1 when not debugging
PATHS = {
    "fineweb-edu-10BT": (
        "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
        f"hf://datasets/{USERNAME}/fineweb-edu-10BT",
    ),
    "slim-pajama-subset-validation": ("data/slim-pajama-subset-validation", "slim-pajama-subset-validation"),
}


# Utility functions
def check_repo(repo_id: str) -> None:
    """Check if HuggingFace repo exists. If not, create it."""
    api = HfApi()
    repo_id = repo_id.replace("hf://datasets/", "")
    logger.info(f"Checking HuggingFace repo {repo_id}")
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(repo_id, repo_type="dataset")
        logger.info(f"Created HuggingFace repo: {repo_id}")
    else:
        logger.info(f"HuggingFace repo already exists: {repo_id}")


def process_with_datatrove(
    source_repo: str, target_repo: str, tokenizer_name_or_path: str, eos_token: str, tok_name: str
) -> None:
    # Step 1. Read and Tokenize. This part of the pipeline is local
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(source_repo, limit=LIMIT),
            DocumentTokenizer(
                local_working_dir=".datatrove/tmp/tokenized/",
                output_folder=".datatrove/scratch/tokenized/",
                save_filename=tok_name,
                max_tokens_per_file=10**10,
                shuffle=False,
                seed=42,
                tokenizer_name_or_path=tokenizer_name_or_path,
                eos_token=eos_token,  # type: ignore
            ),
        ],
        logging_dir=f".datatrove/logs/tokenize_{tok_name}",
    )
    dist_executor.run()

    # Step 2. Merge tokenized files into one big file. Save to HuggingFace Hub
    merge_executor = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=".datatrove/scratch/tokenized/",
                output_folder=f"{target_repo}/{tok_name}",
                save_filename=tok_name,
                max_tokens_per_file=10**10,
                shuffle=False,
                seed=1994,
            )
        ],
        logging_dir=f".datatrove/logs/tokenize_{tok_name}_merge",
        depends=dist_executor,
    )
    merge_executor.run()

    # test reading capabilities
    ds = DatatroveFolderDataset(folder_path=f"{target_repo}/{tok_name}", seq_len=100, shuffle=False)
    logger.info(f"Reading tokenized dataset from HF Hub with {len(ds)=}\n{ds[0]}")

    # remove temp files
    logger.info("Removing temporary folders")
    os.system("rm -rf .datatrove/tmp")
    os.system("rm -rf .datatrove/scratch")


def process_with_datasets(source_repo: str, target_repo: str, tok: PreTrainedTokenizerFast, tok_name: str) -> None:
    ds = load_from_disk(source_repo)
    ds = ds.map(
        lambda ex: tok(ex["text"], return_attention_mask=False, return_token_type_ids=False),
        batched=True,
        num_proc=os.cpu_count() - 1,  # type: ignore
        load_from_cache_file=False,
        desc="Tokenising",
        remove_columns=["text", "meta"],
    )
    logger.info(f"Pushing to HF at {target_repo} at config {tok_name}")
    ds.push_to_hub(target_repo, config_name=tok_name)


# Start
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_tok_path", type=str)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="fineweb-edu-10BT")
    args = parser.parse_args()

    logger.info(f"Tokenizing corpus with tokenizer at {args.raw_tok_path} and {args.vocab_size=}")

    # Load tokenizer and adapt its vocabulary
    raw_tok_path = Path(args.raw_tok_path)
    try:
        logger.info("Tokenizer (transfomers-compatible) already created. Loading it")
        tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            str(raw_tok_path), clean_up_tokenization_spaces=False
        )  # type: ignore

    except OSError:
        assert args.vocab_size is not None
        logger.info(f"Creating tokenizer with {args.vocab_size=} (transfomers-compatible) and loading it")
        tok = load_tokenizer_with_vocab_size(raw_tok_path, args.vocab_size)

    # Save PreTrainedTokenizerFast so its easier to load it from transfomers and datatrove
    tok_name = f"tok-vocab{tok.vocab_size}"
    tok.save_pretrained(str(raw_tok_path / tok_name))  # type: ignore

    # Prepare reading from HF Hub
    source_repo, target_repo = PATHS[args.dataset_name]
    should_use_datatrove = source_repo.startswith("hf://dataset")

    if should_use_datatrove:
        # eos automatically set for training data
        check_repo(target_repo)
        process_with_datatrove(
            source_repo=source_repo,
            target_repo=target_repo,
            tokenizer_name_or_path=str(raw_tok_path / tok_name / "tokenizer.json"),
            eos_token=tok.eos_token,
            tok_name=tok_name,
        )
    else:
        # Here we do not need the eos since it is validation
        process_with_datasets(source_repo=source_repo, target_repo=target_repo, tok=tok, tok_name=tok_name)
