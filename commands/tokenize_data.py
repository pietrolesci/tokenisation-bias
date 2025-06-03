import os
from pathlib import Path

# from datasets import load_from_disk
from datasets import DatasetDict, Features, Sequence, Value, load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,  # type: ignore
)
from typer import Typer

from primer.utilities import get_logger

# Configure the logger and configure colorlog
logger = get_logger("tok-inference", "info")

app = Typer()


def process_minipile(tok: PreTrainedTokenizerFast, tok_name: str) -> None:
    SOURCE_REPO = "JeanKaddour/minipile"
    TARGET_REPO = "pietrolesci/minipile"

    ds: DatasetDict = load_dataset(SOURCE_REPO, cache_dir=".data_cache")  # type: ignore
    ds = ds.map(
        lambda ex: tok(ex["text"], return_attention_mask=False, return_token_type_ids=False),
        batched=True,
        num_proc=os.cpu_count() - 1,  # type: ignore
        load_from_cache_file=False,
        desc="Tokenising",
        remove_columns="text",
        features=Features({"input_ids": Sequence(Value("uint16" if tok.vocab_size < 65535 else "uint32"))}),
    )

    uid = 0
    for split in ds:
        num_rows = len(ds[split])
        ds[split] = ds[split].add_column("uid", list(range(uid, uid + num_rows)))  # type: ignore
        uid += num_rows

    logger.info(f"Pushing to HF at {TARGET_REPO} at config {tok_name}")
    ds.push_to_hub(TARGET_REPO, config_name=tok_name)


def process_finewebedu(tok: PreTrainedTokenizerFast, tok_name: str) -> None:
    SOURCE_REPO = "pietrolesci/finewebedu-20BT"

    ds: DatasetDict = load_dataset(SOURCE_REPO, cache_dir=".data_cache")  # type: ignore

    new_features = {k: v for k, v in ds["train"].features.items() if k == "id"}
    new_features.update({"input_ids": Sequence(Value("uint16" if tok.vocab_size < 65535 else "unit32"))})

    ds = ds.map(
        lambda ex: tok(ex["text"], return_attention_mask=False, return_token_type_ids=False),
        batched=True,
        num_proc=os.cpu_count() - 1,  # type: ignore
        load_from_cache_file=True,
        desc="Tokenising",
        remove_columns=[k for k in ds["train"].column_names if k not in new_features],
        features=Features(new_features),
    )

    uid = 0
    for split in ds:
        num_rows = len(ds[split])
        ds[split] = ds[split].add_column("uid", list(range(uid, uid + num_rows)))  # type: ignore
        uid += num_rows

    logger.info(f"Pushing to HF at {SOURCE_REPO} at config {tok_name}")
    ds.push_to_hub(SOURCE_REPO, config_name=tok_name)


@app.command()
def tokenize(tok_path: str, dataset: str) -> None:
    """Tokenize the dataset using the specified tokenizer."""
    # Load tokenizer and adapt its vocabulary
    path = Path(tok_path)

    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(str(tok_path), clean_up_tokenization_spaces=False)  # type: ignore
    logger.info(f"tok: {tok.vocab_size}, {max(tok.get_vocab().values())}")

    if dataset == "minipile":
        process_minipile(tok, tok_name=path.name)

    elif dataset == "finewebedu":
        process_finewebedu(tok, tok_name=path.name)
