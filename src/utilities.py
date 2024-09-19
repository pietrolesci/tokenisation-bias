# import json
import logging
from typing import Literal

import colorlog

# import srsly
# from transformers import PreTrainedTokenizerFast  # type: ignore

# from tokenizers import Tokenizer


def get_logger(name: str, level: Literal["error", "warning", "info", "debug"] = "info") -> logging.Logger:
    # Convert the level string to the corresponding logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the logger and configure colorlog
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"  # noqa: E501
        )
    )
    logger.addHandler(handler)
    return logger


# def load_tokenizer_with_vocab_size(path: str | Path, vocab_size: int) -> PreTrainedTokenizerFast:
#     path = Path(path)

#     # Edit conf to adapt to the new vocab_size
#     conf: dict = srsly.read_json(path / "tokenizer.json")  # type: ignore

#     # get the number of initial characters used by BPE, these are the things that get merged
#     len_alphabet = len(conf["model"]["vocab"]) - len(conf["model"]["merges"])

#     # the vocab size includes the initial alphabet
#     vocab_sorted_by_token_id = sorted(conf["model"]["vocab"].items(), key=lambda item: item[1])
#     conf["model"]["vocab"] = dict(vocab_sorted_by_token_id[:vocab_size])

#     # but the merges need to be less, such that vocab_size = num_merges + initial_alphabet
#     conf["model"]["merges"] = conf["model"]["merges"][: vocab_size - len_alphabet]

#     # Instantiate tokenizer using tokenizers library
#     backend_tok = Tokenizer.from_str(json.dumps(conf))
#     eos_token: str = srsly.read_yaml(path / "metadata.yaml")["eos_token"]  # type: ignore

#     # Instantiate PreTrainedTokenizerFast from object
#     # NOTE: we do not instantiate from file directly due to compatibility
#     # https://github.com/huggingface/tokenizers/issues/1562#issuecomment-2315349846
#     tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, clean_up_tokenization_spaces=True)

#     # Add common configs for decoder-only models
#     tok.padding_side = "left"  # type: ignore
#     tok.eos_token = eos_token  # type: ignore

#     return tok
