import json
import os
from argparse import ArgumentParser
from pathlib import Path

import srsly
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.utils.dataset import DatatroveFolderDataset
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizerFast  # type: ignore

from src.utilities import get_logger
from tokenizers import Tokenizer

# Configure the logger and configure colorlog
logger = get_logger("tok-inference", "info")


# Global options
HF_URL = "hf://datasets"
USERNAME = "pietrolesci"


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


def load_tokenizer_with_vocab_size(path: str | Path, vocab_size: int) -> PreTrainedTokenizerFast:
    path = Path(path)

    # Edit conf to adapt to the new vocab_size
    conf: dict = srsly.read_json(path / "tokenizer.json")  # type: ignore

    # get the number of initial characters used by BPE, these are the things that get merged
    len_alphabet = len(conf["model"]["vocab"]) - len(conf["model"]["merges"])

    logger.info(
        f"Creating tokenizer with {vocab_size=}, with initial alphabet of {len_alphabet} characters"
        f"and {vocab_size - len_alphabet} merges"
    )

    # the vocab size includes the initial alphabet
    conf["model"]["vocab"] = dict(list(conf["model"]["vocab"].items())[:vocab_size])

    # but the merges need to be less, such that vocab_size = num_merges + initial_alphabet
    conf["model"]["merges"] = conf["model"]["merges"][: vocab_size - len_alphabet]

    # Instantiate tokenizer using tokenizers library
    backend_tok = Tokenizer.from_str(json.dumps(conf))
    eos_token: str = srsly.read_yaml(path / "metadata.yaml")["eos_token"]  # type: ignore

    # Instantiate PreTrainedTokenizerFast from object
    # NOTE: we do not instantiate from file directly due to compatibility
    # https://github.com/huggingface/tokenizers/issues/1562#issuecomment-2315349846
    tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok)

    # Add common configs for decoder-only models
    logger.info(f"Setting EOS token to {eos_token} and padding_size to `left`")
    tok.padding_side = "left"  # type: ignore
    tok.eos_token = eos_token  # type: ignore

    return tok


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_tok_path", type=str)
    parser.add_argument("--vocab_size", type=int)
    args = parser.parse_args()

    # Load tokenizer and adapt its vocabulary
    raw_tok_path = Path(args.raw_tok_path)
    tok = load_tokenizer_with_vocab_size(raw_tok_path, args.vocab_size)

    # Save PreTrainedTokenizerFast so its easier to load it from transfomers and datatrove
    tok_name = f"tok-vocab{args.vocab_size}"
    tok.save_pretrained(str(raw_tok_path / tok_name))  # type: ignore

    # Prepare reading from HF Hub
    source_repo = "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"
    target_repo = f"hf://datasets/{USERNAME}/fineweb-edu-10BT"
    check_repo(target_repo)

    # Step 1. Read and Tokenize. This part of the pipeline is local
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(source_repo, limit=10000),  # NOTE: remove limit when not debugging
            DocumentTokenizer(
                local_working_dir=".datatrove/tmp/tokenized/",
                output_folder=".datatrove/scratch/tokenized/",
                save_filename=tok_name,
                max_tokens_per_file=10**10,
                shuffle=False,
                seed=42,
                tokenizer_name_or_path=str(raw_tok_path / tok_name / "tokenizer.json"),
                eos_token=tok.eos_token,  # type: ignore
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
    logger.info("Reading tokenized dataset from HF Hub\nLength", len(ds[0]["input_ids"]), ds[0])

    # remove temp files
    logger.info("Removing temporary folders")
    os.system("rm -rf .datatrove/tmp")
    os.system("rm -rf .datatrove/scratch")
