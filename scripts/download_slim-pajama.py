import shutil
from pathlib import Path

import polars as pl
from huggingface_hub import snapshot_download

from src.utilities import get_logger

logger = get_logger("slim-pajama")

LOCAL_DIR = Path("data/slim-pajama-eval")

if __name__ == "__main__":
    logger.info("Cloning repo from HF")
    snapshot_download(
        "cerebras/SlimPajama-627B",
        revision="refs/convert/parquet",
        repo_type="dataset",
        local_dir=str(LOCAL_DIR / "tmp"),
        allow_patterns=["default/partial-validation/*", "default/partial-test/*"],
    )

    logger.info("Removing {'RedPajamaGithub', 'RedPajamaStackExchange'} fields.")
    val_df = pl.scan_parquet(str(LOCAL_DIR / "tmp" / "default" / "partial-validation" / "*.parquet")).with_columns(
        split=pl.lit("validation")
    )
    test_df = pl.scan_parquet(str(LOCAL_DIR / "tmp" / "default" / "partial-test" / "*.parquet")).with_columns(
        split=pl.lit("test")
    )

    df = (
        pl.concat([val_df, test_df])
        .with_columns(meta=pl.col("meta").struct.field("redpajama_set_name"))
        .with_row_index("uid")
        .filter(pl.col("meta").is_in(["RedPajamaGithub", "RedPajamaStackExchange"]).not_())
        .collect()
    )

    logger.info(f"Saving to {LOCAL_DIR=}")
    df.write_parquet(str(LOCAL_DIR) + ".parquet")

    # logger.info("Converting to HF dataset")
    # ds = Dataset.from_polars(df)
    # ds.save_to_disk(LOCAL_DIR)

    # logger.info(f"Pushing to HF at {LOCAL_DIR.name}")
    # ds.push_to_hub(LOCAL_DIR.name, config_name="default")

    logger.info("Removing useless folders")
    shutil.rmtree(LOCAL_DIR / ".cache", ignore_errors=True)
    shutil.rmtree(LOCAL_DIR / "default", ignore_errors=True)
    shutil.rmtree(LOCAL_DIR / "tmp", ignore_errors=True)
