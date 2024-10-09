from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger

from src.dataset import TokenizedDocumentDataset
from src.utilities import get_logger

# Configure the logger and configure colorlog
logger = get_logger("trial", "info")


if __name__ == "__main__":
    tokenizer_name_or_path = "/home/pl487/rdd/outputs/tokenizers/bpe32000minipile"
    eos_token = None
    data_path = "/home/pl487/rdd/data/"

    # Step 1. Read and Tokenize. This part of the pipeline is local
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_path,
                file_progress=False,
                doc_progress=True,
                shuffle_files=False,
                text_key="input_ids",
                id_key="uid",
                limit=-1,
                glob_pattern="minipile-eval-bpe32000minipile.parquet",
            ),
            TokenizedDocumentDataset(
                output_folder=".datatrove/scratch/tokenized/",
                local_working_dir=".datatrove/tmp/tokenized/",
                save_filename="minipile-eval-bpe32000minipile",
                max_tokens_per_file=10**10,
                shuffle=False,
                tokenizer_name_or_path=f"{tokenizer_name_or_path}/tokenizer.json",
                eos_token=eos_token,  # type: ignore
            ),
        ],
        logging_dir=".datatrove/logs/trial",
    )
    dist_executor.run()

    # Step 2. Merge tokenized files into one big file. Save to HuggingFace Hub
    merge_executor = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=".datatrove/scratch/tokenized/",
                output_folder=f"{data_path}/minipile-eval",
                save_filename="minipile-eval-bpe32000minipile",
                max_tokens_per_file=10**10,
                shuffle=False,
                # seed=1994,
            )
        ],
        logging_dir=".datatrove/logs/trial_merge",
        depends=dist_executor,
    )
    merge_executor.run()
