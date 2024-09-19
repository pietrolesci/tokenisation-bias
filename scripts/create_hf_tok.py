import json
from argparse import ArgumentParser
from pathlib import Path

import srsly
from transformers import PreTrainedTokenizerFast  # type: ignore

from src.utilities import get_logger
from tokenizers import Tokenizer

# Configure the logger and configure colorlog
logger = get_logger("tok_creation", "info")

# Global options
DEFAULT_DIR = "outputs/tokenizers"
VOCAB_SIZES = [128 * 63, 128 * 125, 128 * 250, 128 * 500, 128 * 1000, 128 * 2000]


def load_tokenizer_with_vocab_size(path: str | Path, vocab_size: int) -> PreTrainedTokenizerFast:
    path = Path(path)

    # Edit conf to adapt to the new vocab_size
    conf: dict = srsly.read_json(path / "tokenizer.json")  # type: ignore

    # get the number of initial characters used by BPE, these are the things that get merged
    len_alphabet = len(conf["model"]["vocab"]) - len(conf["model"]["merges"])

    # the vocab size includes the initial alphabet
    vocab_sorted_by_token_id = sorted(conf["model"]["vocab"].items(), key=lambda item: item[1])
    conf["model"]["vocab"] = dict(vocab_sorted_by_token_id[:vocab_size])

    # but the merges need to be less, such that vocab_size = num_merges + initial_alphabet
    conf["model"]["merges"] = conf["model"]["merges"][: vocab_size - len_alphabet]

    # Instantiate tokenizer using tokenizers library
    backend_tok = Tokenizer.from_str(json.dumps(conf))
    eos_token: str = srsly.read_yaml(path / "metadata.yaml")["eos_token"]  # type: ignore

    # Instantiate PreTrainedTokenizerFast from object
    # NOTE: we do not instantiate from file directly due to compatibility
    # https://github.com/huggingface/tokenizers/issues/1562#issuecomment-2315349846
    tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, clean_up_tokenization_spaces=True)

    # Add common configs for decoder-only models
    tok.padding_side = "left"  # type: ignore
    tok.eos_token = eos_token  # type: ignore

    return tok


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_tok_path", type=str)
    args = parser.parse_args()

    tok_path = Path(args.raw_tok_path)
    tok_type = tok_path.name.split("_")[0]  # should be bpe
    logger.info(f"Creating tokenizer at {DEFAULT_DIR}")

    for vocab_size in VOCAB_SIZES:
        out_path = Path(DEFAULT_DIR) / f"{tok_type}{vocab_size}"
        logger.info(f"Saving with vocab {vocab_size=} at {out_path=}")
        tok = load_tokenizer_with_vocab_size(tok_path, vocab_size=vocab_size)
        tok.save_pretrained(str(out_path))
        with (out_path / "raw_tok_path.txt").open(mode="w") as fl:
            fl.write(str(tok_path.absolute()))
