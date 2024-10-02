from lightning import seed_everything
from transformers import PreTrainedTokenizerFast
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

MODEL_TYPE = LlamaForCausalLM | GPTNeoXForCausalLM | GPT2LMHeadModel


def get_model(name: str, tok: PreTrainedTokenizerFast) -> tuple[MODEL_TYPE, PretrainedConfig]:
    kwargs = {
        "vocab_size": tok.vocab_size,
        "bos_token_id": tok.eos_token_id,  # type: ignore
        "eos_token_id": tok.eos_token_id,  # type: ignore
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "max_position_embeddings": 2048,
        # _attn_implementation="flash_attention_2",
    }
    seed_everything(42)
    if name == "hermes":
        # adapted from https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/blob/main/config.json
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=1024,
            intermediate_size=2 * 1024,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=8,
            tie_word_embeddings=False,
            initializer_range=0.02,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            # rope_scaling={
            #     "factor": 8.0,
            #     "high_freq_factor": 4.0,
            #     "low_freq_factor": 1.0,
            #     "original_max_position_embeddings": 2048,
            #     "rope_type": "llama3",
            # },
            # rope_theta=500000.0,
            **kwargs,
        )
        model = LlamaForCausalLM(config)

    elif name == "smol_llama-101m-gqa":
        # https://huggingface.co/BEE-spoke-data/smol_llama-101M-GQA/blob/main/config.json
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=24,
            num_key_value_heads=8,
            num_hidden_layers=6,
            tie_word_embeddings=False,
            initializer_range=0.02,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000.0,
            **kwargs,
        )
        model = LlamaForCausalLM(config)

    elif name == "smol_llama-81M-tied":  
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=24,
            num_key_value_heads=24,
            num_hidden_layers=6,
            tie_word_embeddings=True,
            initializer_range=0.02,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000.0,
            **kwargs,
        )
        model = LlamaForCausalLM(config)

    elif name == "smollm-135m":
        # adapted from SmolLM https://huggingface.co/HuggingFaceTB/SmolLM-135M/blob/main/config.json
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=576,
            intermediate_size=1536,
            num_attention_heads=9,
            num_key_value_heads=3,
            num_hidden_layers=30,
            tie_word_embeddings=True,
            initializer_range=0.02,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000.0,
            **kwargs,
        )
        model = LlamaForCausalLM(config)

    elif name == "pythia-14m":
        # https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json
        config = GPTNeoXConfig(
            model_type="gpt_neox",
            hidden_act="gelu",
            hidden_size=128,
            intermediate_size=512,
            num_attention_heads=4,
            num_hidden_layers=6,
            tie_word_embeddings=False,
            initializer_range=0.02,
            classifier_dropout=0.1,
            layer_norm_eps=1e-05,
            rotary_emb_base=10000,
            rotary_pct=0.25,
            use_parallel_residual=True,
            **kwargs,
        )
        model = GPTNeoXForCausalLM(config)

    elif name == "pythia-31m":
        # https://huggingface.co/EleutherAI/pythia-31m/blob/main/config.json
        config = GPTNeoXConfig(
            model_type="gpt_neox",
            hidden_act="gelu",
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=8,
            num_hidden_layers=6,
            tie_word_embeddings=False,
            initializer_range=0.02,
            classifier_dropout=0.1,
            layer_norm_eps=1e-05,
            rotary_emb_base=10000,
            rotary_pct=0.25,
            use_parallel_residual=True,
            **kwargs,
        )
        model = GPTNeoXForCausalLM(config)

    elif name == "gpt2":
        config = GPT2Config(
            model_type="gpt2",
            activation_function="gelu_new",
            n_embd=768,
            n_ctx=kwargs["max_position_embeddings"],
            n_positions=kwargs["max_position_embeddings"],
            n_head=12,
            n_layer=12,
            initializer_range=0.02,
            attn_pdrop=0.1,
            embd_pdrop=0.1,
            layer_norm_epsilon=1e-05,
            resid_pdrop=0.1,
            summary_activation=None,
            summary_first_dropout=0.1,
            summary_proj_to_labels=True,
            summary_type="cls_index",
            summary_use_proj=True,
            **kwargs,
        )
        model = GPT2LMHeadModel._from_config(config)

    else:
        raise ValueError

    return model, config
