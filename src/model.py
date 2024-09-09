import torch
from transformers import PreTrainedTokenizerFast, set_seed
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def get_model(name: str, tok: PreTrainedTokenizerFast) -> tuple[torch.nn.Module, PretrainedConfig]:
    kwargs = {
        "vocab_size": tok.vocab_size,
        "bos_token_id": tok.eos_token_id,  # type: ignore
        "eos_token_id": tok.eos_token_id,  # type: ignore
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "max_position_embeddings": 2048,
        # _attn_implementation="flash_attention_2",
    }
    set_seed(42)
    if name == "smollm":
        # adapted from SmolLM https://huggingface.co/HuggingFaceTB/SmolLM-135M/blob/main/config.json
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=9,
            num_key_value_heads=3,
            num_hidden_layers=8,
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

    elif name == "pythia":
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

    else:
        raise ValueError

    return model, config
