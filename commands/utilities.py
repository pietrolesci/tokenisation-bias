import json
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import srsly
import typer
import yaml
from huggingface_hub import HfApi, logging
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,  # type: ignore
)
from typer import Typer

from primer.model import load_hf_from_pl
from primer.utilities import get_logger
from tokenizers import Tokenizer

logger = get_logger("utilities")

app = Typer()


# ================================
# Default paths and configurations
# ================================
MODEL_EVAL_PATH = "./data/me-minipile-evals"
REGDATA_PATH = "./data/me-minipile-regdata"
TOK_PATH = "outputs/tokenizers"
MODEL_NAME_USERNAME = "pietrolesci"
EVALS_TARGET_REPO = "pietrolesci/me-minipile-evals"


# ====================================================
# Export evaluation results to a unique parquet files.
# ====================================================
@app.command()
def export_evals(eval_path: str = "./outputs/multirun/model_eval/minipile", out_path: str = MODEL_EVAL_PATH) -> None:
    """Export evaluation results to parquet files."""
    path = Path(eval_path)
    local_path = Path(out_path)
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"{local_path} created")

    for p in path.iterdir():
        filename = f"{p.name.split('_202')[0]}.parquet"
        dest = local_path / filename
        logger.info(f"Working on {p} and saving to {dest}")
        (
            pl.scan_parquet(f"{eval_path}/{p.name}/*/*.parquet", include_file_paths="filename")
            .with_columns(step=pl.col("filename").str.extract("step(\\d+)").cast(pl.Int64))
            .drop("filename")
            .sink_parquet(local_path / filename)
        )


@app.command()
def upload_evals(
    eval_path: str = MODEL_EVAL_PATH, regdata_path: str = REGDATA_PATH, repo_id: str = EVALS_TARGET_REPO
) -> None:
    """Upload evaluation results to a Hugging Face dataset repository."""
    path = Path(eval_path)
    assert path.exists(), f"Path {path} does not exist."

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="dataset")
    # api.upload_folder(folder_path=path, repo_id=repo_id, repo_type="dataset", revision="main", path_in_repo="evals")

    if regdata_path:
        path = Path(regdata_path)
        assert path.exists(), f"Path {path} does not exist."
        api.upload_folder(
            folder_path=path, repo_id=repo_id, repo_type="dataset", revision="main", path_in_repo="reg_data"
        )


# ===================================================================
# Convert tokenizers to `transformers.PreTrainedTokenizerFast` format
# ===================================================================
# VOCAB_SIZES = [128 * 63, 128 * 125, 128 * 250, 128 * 500, 128 * 1000, 128 * 2000]


def bpe2wp(conf: dict) -> dict:
    # Super hacky way to make a BPE tokenizer work with WordPiece
    conf["model"]["type"] = "WordPiece"
    conf["model"]["unk_token"] = conf["added_tokens"][0]["content"]
    conf["model"]["continuing_subword_prefix"] = ""
    conf["model"]["max_input_chars_per_word"] = 100
    return conf


def load_tokenizer_with_vocab_size(
    path: str | Path, vocab_size: int, convertion: Literal["bpe2wp", "wp2bpe"] | None
) -> PreTrainedTokenizerFast:
    path = Path(path)

    # Edit conf to adapt to the new vocab_size
    conf: dict = srsly.read_json(path / "tokenizer.json")  # type: ignore

    if conf["model"]["type"].lower() == "bpe":
        # get the number of initial characters used by BPE, these are the things that get merged
        len_alphabet = len(conf["model"]["vocab"]) - len(conf["model"]["merges"])

        # but the merges need to be less, such that vocab_size = num_merges + initial_alphabet
        conf["model"]["merges"] = conf["model"]["merges"][: vocab_size - len_alphabet]

    # the vocab size includes the initial alphabet
    vocab_sorted_by_token_id = sorted(conf["model"]["vocab"].items(), key=lambda item: item[1])
    conf["model"]["vocab"] = dict(vocab_sorted_by_token_id[:vocab_size])

    # convert to another tokenizer
    if convertion:
        if convertion == "bpe2wp":
            assert conf["model"]["type"].lower() == "bpe"
            conf = bpe2wp(conf)
        elif convertion == "wp2bpe":
            raise NotImplementedError("wp2bpe conversion is not implemented")

    # Instantiate tokenizer using tokenizers library
    backend_tok = Tokenizer.from_str(json.dumps(conf))
    meta = srsly.read_yaml(path / "metadata.yaml")
    eos_token: str = meta["eos_token"]  # type: ignore
    unk_token: str | None = meta.get("unk_token", None)  # type: ignore

    # Instantiate PreTrainedTokenizerFast from object
    # NOTE: we do not instantiate from file directly due to compatibility
    # https://github.com/huggingface/tokenizers/issues/1562#issuecomment-2315349846
    tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, clean_up_tokenization_spaces=True)

    # Add common configs for decoder-only models
    tok.padding_side = "left"  # type: ignore
    tok.eos_token = eos_token  # type: ignore
    tok.unk_token = unk_token  # type: ignore

    return tok


@app.command()
def convert_tokenizer(raw_tok_path: str, vocab_size: int, convertion: str | None = None) -> None:
    tok_path = Path(raw_tok_path)
    names = tok_path.name.split("_")

    tok_type = names[0]  # should be bpe
    dataset = names[1] if len(names) > 2 else ""

    logger.info(f"Creating tokenizer at {TOK_PATH}")

    out_path = Path(TOK_PATH) / f"{convertion or tok_type}{vocab_size}{dataset}"
    logger.info(f"Saving with vocab {vocab_size=} at {out_path=}")
    tok = load_tokenizer_with_vocab_size(tok_path, vocab_size=vocab_size, convertion=convertion)  # type: ignore
    tok.save_pretrained(str(out_path))
    with (out_path / "raw_tok_path.txt").open(mode="w") as fl:
        fl.write(str(tok_path.absolute()))


# =======================================================
# Upload model and evaluation results to Hugging Face Hub
# =======================================================
logging.set_verbosity_info()  # or _debug for more info


@app.command()
def upload_model(run_dir: Annotated[str, typer.Argument(help="Path to the directory of the training run.")]) -> None:
    run_path = Path(run_dir)

    logger.info("read hparams")
    hparams: dict = srsly.read_yaml(run_path / "hparams.yaml")  # type: ignore

    logger.info("write readme")
    lines = f"## Experiment Configuration\n```yaml\n{yaml.dump(hparams)}```".replace("/home/pl487", ".")
    with (run_path / "README.md").open("w") as fl:
        fl.writelines(lines)

    logger.info("create repo and upload common files")
    repo_id = f"{MODEL_NAME_USERNAME}/{hparams['model']}_{hparams['dataset']}_{hparams['tok_name']}"

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=run_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".hydra/*", ".checkpoints/*", "*.log", "*.err", "*.out"],  # Ignore all text logs
        revision="main",
    )

    logger.info("upload checkpoints to different branches")
    branches = [branch.name for branch in api.list_repo_refs(repo_id).branches]
    for p in (run_path / ".checkpoints").iterdir():
        if "last" in p.name or p.stem in branches:
            continue
        logger.info(f"Uploading {p.stem}")
        ckpt = load_hf_from_pl(p)
        ckpt.push_to_hub(repo_id, revision=p.stem)  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(f"./outputs/tokenizers/{hparams['tok_name']}")
    tokenizer.push_to_hub(repo_id, revision="main")
