import os
from argparse import ArgumentParser
from pathlib import Path

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.utils.dataset import DatatroveFolderDataset
from huggingface_hub import HfApi

from src.utilities import get_logger, load_tokenizer_with_vocab_size

# Configure the logger and configure colorlog
logger = get_logger("tok-inference", "info")


# Global options
HF_URL = "hf://datasets"
USERNAME = "pietrolesci"
LIMIT = -1  # NOTE: set to -1 when not debugging

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_tok_path", type=str)
    parser.add_argument("--vocab_size", type=int)
    args = parser.parse_args()

    logger.info(f"Tokenizing corpus with tokenizer at {args.raw_tok_path} and {args.vocab_size=}")

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
            ParquetReader(source_repo, limit=LIMIT), 
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
    logger.info(f"Reading tokenized dataset from HF Hub with {len(ds)=}\n{ds[0]}")

    # remove temp files
    logger.info("Removing temporary folders")
    os.system("rm -rf .datatrove/tmp")
    os.system("rm -rf .datatrove/scratch")
