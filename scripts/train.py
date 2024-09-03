import os
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedTokenizerFast, set_seed
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.training_args import TrainingArguments

from src.trainer import LMTrainer
from src.utilities import get_logger

# Configure the logger and configure colorlog
logger = get_logger("training", "info")
TRAINING_ARGS_NAME = "training_args.bin"

api = HfApi()

if __name__ == "__main__":
    tok_path = Path("/home/pl487/rdd/outputs/tokenizer_train/2024-08-30T12-00-43/tok-vocab32000")
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(str(tok_path))  # type: ignore
    tok.eos_token_id = 0
    tok.pad_token_id = 0

    # Adapted from SmolLM
    # https://huggingface.co/HuggingFaceTB/SmolLM-135M/blob/main/config.json
    config = LlamaConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        hidden_act="silu",
        hidden_size=512,
        intermediate_size=1024,
        initializer_range=0.02,
        max_position_embeddings=2048,
        mlp_bias=False,
        model_type="llama",
        num_attention_heads=9,
        num_hidden_layers=8,
        num_key_value_heads=3,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        use_cache=True,
        vocab_size=tok.vocab_size,
    )

    set_seed(42)
    model = LlamaForCausalLM(config)
    model_name = f"SmolLM-{model.num_parameters() / 1e6:.0f}M-tok{tok.vocab_size}"
    model.push_to_hub(model_name, revision="init")  # type: ignore
    model.save_pretrained(f"outputs/model_training/{model_name}/checkpoint-0")
    
    # model.push_to_hub(model_name, revision="init")  # type: ignore
    # logger.info(f"Pushing model to hub at {model_name} repo")
    logger.info(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    logger.info(f"Num parameters: {model.num_parameters() / 1e6:.1f}M")

    # too many arguments, use the set methods to make things clearer
    training_args = TrainingArguments(
        # =======
        # logging
        # =======
        output_dir=f"outputs/model_training/{model_name}",
        logging_dir=f"outputs/model_training/{model_name}/tb_logs",
        logging_strategy="steps",
        logging_first_step=True,
        log_level="passive",  # takes it from global
        logging_steps=1,
        report_to="tensorboard",
        include_num_input_tokens_seen=True,
        # =============
        # checkpointing
        # =============
        save_strategy="steps",
        save_steps=500,
        save_safetensors=True,
        # ===========
        # push to hub
        # ===========
        # push_to_hub=True,
        # hub_model_id=model_name,
        # hub_strategy="all_checkpoints",
        # hub_private_repo=True,
        # =====
        # setup
        # =====
        eval_strategy="no",
        seed=42,
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        torch_compile=True,
        # ============
        # optimisation
        # ============
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        learning_rate=2e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        # lr_scheduler_type="wsd",
        lr_scheduler_kwargs=dict(
            final_lr_factor=0.0, init_div_factor=100, frac_decay=0.2, decay_type="sqrt"
        ),  # use to pass
        warmup_steps=2_000,
        num_train_epochs=1,
        max_steps=300_000,
        # ===========
        # dataloading
        # ===========
        dataloader_num_workers=os.cpu_count() - 1,  # type: ignore
    )

    set_seed(training_args.seed)
    target_repo = "hf://datasets/pietrolesci/fineweb-edu-10BT"
    trainer = LMTrainer(
        model, args=training_args, tokenizer=tok, config=config, data_path=f"{target_repo}/{tok_path.name}"
    )
    trainer.train()
