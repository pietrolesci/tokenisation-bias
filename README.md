# rdd


## Setup

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# create environment
uv sync  # <- check this is enough
```

## Train tokenizer

First train a tokenizer (with a large vocabulary) on fineweb-edu-10BT

```bash
uv run scripts/tok_train.py --tok_type <bpe or wordpiece>
```

This will save the tokenizer and useful metadata into a default folder named `{tok_type}_{"%Y-%m-%dT%H-%M-%S"}` under the default directory `outputs/tok_train`

We use a patched version of HuggingFace `tokenizers` library that saves information about merges and their counts. In this way, we can train the tokenizer once and create new tokenizer of arbitrary vocabulary size (of course, smaller than the maximum vocabulary size we used to train the tokenizer). So, we create tokenizers (and save in a way that makes it easily load-able with the `AutoTokenizer` class) of different sizes as follow:

```bash
uv run scripts/create_hf_tok.py --raw_tok_path outputs/tok_train/<tok_type}_{"%Y-%m-%dT%H-%M-%S"}>
```

This saves the `AutoTokenizer`-compliant tokenizers in the default directory `outputs/tokenizers` and each tokenizer is a folder named `{tok_type}{vocab_size}`, where `tok_type` is automatically fetched from the parent tokenizer while `vocab_size` is hard-coded (we use 8k, 16k, 32k, 64k, 128k, 256k).


## Tokenize corpus

Once we have the tokenizers, we tokenize the fineweb-edu-10BT corpus. This streams the corpus from the HuggingFace Hub and saves it again to the Hub to save space on the local disk.[Add more details on the format of the output file and what info is required, eg login, to read and write to HF Hub] 

[Add how to tokenize validation corpus]

```bash
# used for training
uv run scripts/tokenize_data.py --tok_path outputs/tokenizers/<{tok_type}{vocab_size}> --dataset_name fineweb-edu-10BT

# used for validation
uv run scripts/tokenize_data.py --tok_path outputs/tokenizers/<{tok_type}{vocab_size}> --dataset_name slim-pajama-subset-validation
```

On the HuggingFace Hub, we create a repo for each dataset and within that repo we have a folder for each tokenizer named `{hf_repo_id}/{tok_type}{vocab_size}`. There is a special folder `{hf_repo_id}/{data}` that contains the original data.


## Train model

This will save model artefacts under the default directory `outputs/model_train` into a folder named `{model}-{num_params}-{tok_type}{vocab_size}`

```bash
uv run scripts/model_train.py <overwrite configs using hydra cli>
```


## Filter the tokens to compute the likelihood for

[This is a notebook for now]

Creates a folder under `data` called `slim-pajama-subset-validation-sample-{tok_type}{vocab_size}`


# Compute token surprisal

```bash
uv run scripts/model_eval.py
```



