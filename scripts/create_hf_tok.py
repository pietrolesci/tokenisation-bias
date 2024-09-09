from argparse import ArgumentParser
from pathlib import Path

from src.utilities import get_logger, load_tokenizer_with_vocab_size

# Configure the logger and configure colorlog
logger = get_logger("tok_creation", "info")

# Global options
DEFAULT_DIR = "outputs/tokenizers"
VOCAB_SIZES = [128 * 63, 128 * 125, ]#128 * 250, 128 * 500, 128 * 1000, 128 * 2000]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_tok_path", type=str)
    args = parser.parse_args()

    tok_path = Path(args.raw_tok_path)
    tok_type = tok_path.name.split("_")[0]  # should be bpe
    logger.info(f"Creating tokenizer at {DEFAULT_DIR}")

    for vocab_size in VOCAB_SIZES:
        out_path = str(Path(DEFAULT_DIR) / f"{tok_type}{vocab_size}")
        logger.info(f"Saving with vocab {vocab_size=} at {out_path=}")
        tok = load_tokenizer_with_vocab_size(tok_path, vocab_size=vocab_size)
        tok.save_pretrained(str(out_path))
