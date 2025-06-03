import datetime
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Literal

import srsly
import typer
from datasets import IterableDataset, load_dataset

from primer.utilities import get_logger, track_time
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

app = typer.Typer()

# Configure the logger and configure colorlog
logger = get_logger("tok-train", "info")

# Global options
MAX_VOCAB_SIZE = 320_000  # 128 * 2500
EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"


def load_finewebedu(cache_dir: str) -> IterableDataset:
    return load_dataset("pietrolesci/finewebedu-20BT", "default", split="train", streaming=True, cache_dir=cache_dir)  # type: ignore


def load_minipile(cache_dir: str) -> IterableDataset:
    return load_dataset("JeanKaddour/minipile", split="train", streaming=True, cache_dir=cache_dir)  # type: ignore


NAME_TO_DATASET_AND_LEN = {
    "finewebedu": (load_finewebedu, 20_000_000),
    "minipile": (load_minipile, 1_000_000),  # all of them
}


def get_tok(
    tok_type: Literal["bpe", "wordpiece", "unigramlm"], vocab_size: int = MAX_VOCAB_SIZE
) -> tuple[Tokenizer, trainers.BpeTrainer | trainers.WordPieceTrainer, trainers.UnigramTrainer]:
    # Define the tokenizer and set up the tokenizer components (this is a GPT-2 tokenizer)
    # https://github.com/huggingface/tokenizers/blob/14a07b06e4a8bd8f80d884419ae4630f5a3d8098/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py#L10
    mapping = {
        "bpe": (models.BPE, trainers.BpeTrainer),
        "wordpiece": (partial(models.WordPiece, unk_token=UNK_TOKEN), trainers.WordPieceTrainer),
        "unigramlm": (models.Unigram, trainers.UnigramTrainer),
    }
    model_cls, trainer_cls = mapping[tok_type]

    # Tokenizer
    tokenizer = Tokenizer(model_cls())  # type: ignore
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)  # type: ignore
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)  # type: ignore
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore

    # Trainer
    kwargs = {
        "vocab_size": vocab_size,
        "special_tokens": [EOS_TOKEN] if tok_type == "bpe" else [EOS_TOKEN, UNK_TOKEN],
        "show_progress": True,
        "initial_alphabet": pre_tokenizers.ByteLevel.alphabet(),
    }
    if tok_type != "unigramlm":
        kwargs["min_frequency"] = 2
    else:
        # weird that here we need to pass UNK_TOKEN to the trainer: https://github.com/huggingface/tokenizers/issues/1779
        kwargs["unk_token"] = UNK_TOKEN
        kwargs["shrinking_factor"] = 0.75  # these are the defaults
        kwargs["n_sub_iterations"] = 2  # these are the defaults

    trainer = trainer_cls(**kwargs)

    return tokenizer, trainer  # type: ignore


@app.command()
def train(
    tok_type: str = "bpe",
    dataset: str = "minipile",
    cache_dir: str = ".data_cache",
    batch_size: int = 1_000,
    out_path: str = "./outputs/tok_train/",
) -> None:
    # Define the trainer and tokenizer
    tokenizer, trainer = get_tok(tok_type)  # type: ignore
    logger.info(f"{tokenizer=}, {trainer=}")

    load_fn, num_docs = NAME_TO_DATASET_AND_LEN[dataset]
    logger.info(f"Streaming {num_docs=} from dataset={dataset}")
    dataset = load_fn(cache_dir)  # type: ignore
    dataset = dataset.take(num_docs).select_columns(["text"]).batch(batch_size=batch_size)  # type: ignore

    # Train
    with track_time(f"Training tokenizer with {MAX_VOCAB_SIZE=} on {num_docs=}"):
        tokenizer.train_from_iterator(iter(x["text"] for x in dataset), trainer, num_docs)  # type: ignore

    # Save
    logger.info("Saving")
    tokenizer.save("tokenizer.json", pretty=True)

    # Tidy and put the artefacts of this run into its own folder
    logger.info("Waiting for all files to be written")
    filenames = ["tokenizer.json", "all_merges.jsonl", "implemented_merges.jsonl"]
    while not all(Path(fl).exists() for fl in filenames):
        time.sleep(1)
    logger.info("All files written!")

    path = Path(out_path)
    folder_path = path / (f"{tok_type}_{dataset}_" + datetime.datetime.now().strftime(r"%Y-%m-%dT%H-%M-%S"))
    folder_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Tidying: Moving files to {folder_path=}")

    for fl in filenames:
        shutil.move(fl, folder_path / fl)

    meta = {"max_vocab_size": MAX_VOCAB_SIZE, "num_docs": num_docs, "eos_token": EOS_TOKEN}
    if tok_type == "wordpiece":
        meta["unk_token"] = UNK_TOKEN
    srsly.write_yaml(folder_path / "metadata.yaml", meta)
