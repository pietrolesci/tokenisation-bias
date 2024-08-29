import datetime
import shutil
import time
from pathlib import Path

import srsly
from datasets import IterableDataset, load_dataset

from src.utilities import get_logger
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

# Configure the logger and configure colorlog
logger = get_logger("tok-train", "info")

# Global options
MAX_VOCAB_SIZE = 128 * 2000
EOS_TOKEN = "<|endoftext|>"
NUM_DOCS = 9_000_000  # out of 9.67M
BATCH_SIZE = 10_000

if __name__ == "__main__":
    start_time = time.time()

    logger.info(f"Training tokenizer with vocab size {MAX_VOCAB_SIZE} on dataset with {NUM_DOCS} docs")

    # Define the trainer
    trainer = BpeTrainer(vocab_size=MAX_VOCAB_SIZE, min_frequency=2, special_tokens=[EOS_TOKEN], show_progress=True)  # type: ignore
    logger.info(f"Trainer: {trainer}")

    # Define the tokenizer and set up the tokenizer components (this is a GPT-2 tokenizer)
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)  # type: ignore
    tokenizer.post_processor = ByteLevelProcessor()  # type: ignore
    tokenizer.decoder = ByteLevelDecoder()  # type: ignore
    logger.info(f"Tokenizer: {tokenizer}")

    # Stream the fineweb-edu-10BT dataset from the HF Hub
    logger.info("Preparing to stream the dataset")
    dataset: IterableDataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True, cache_dir=".data_cache"
    )  # type: ignore
    dataset = dataset.take(NUM_DOCS).select_columns(["text"]).batch(batch_size=BATCH_SIZE)

    # Train
    logger.info("Starting training")
    tokenizer.train_from_iterator(iter(x["text"] for x in dataset), trainer, NUM_DOCS)
    logger.info("Training done!")

    # Save
    logger.info("Saving")
    tokenizer.save("tokenizer.json", pretty=True)

    # Tidy and put the artefacts of this run into its own folder
    logger.info("Waiting for all files to be written")
    filenames = ["tokenizer.json", "all_merges.jsonl", "implemented_merges.jsonl"]
    while not all(Path(fl).exists() for fl in filenames):
        time.sleep(1)
    logger.info("All files written!")

    logger.info("Tidying: Moving files to folder")
    path = Path("./outputs/tokenizers/")
    folder_path = path / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    folder_path.mkdir(parents=True, exist_ok=True)
    for fl in filenames:
        shutil.move(fl, folder_path / fl)

    srsly.write_yaml(
        folder_path / "metadata.yaml", {"max_vocab_size": MAX_VOCAB_SIZE, "num_docs": NUM_DOCS, "eos_token": EOS_TOKEN}
    )

    # Compute the total runtime
    logger.info(f"Total runtime: {int((start_time - time.time()) / 60)} minutes")
