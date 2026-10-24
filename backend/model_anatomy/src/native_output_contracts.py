"""Narrow output contracts for reviewed native Transformers modules.

These describe the canonical eval, no-cache, no-autocast profile. They inspect
module structure and parameter metadata only; no forward method is called.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
from pathlib import Path

import torch
from torch import nn

from .ir import TensorSpec

# SHA256 of the reviewed installed source files. A different upstream source
# version must be reviewed before these contracts can be used again.
CONTRACT_SOURCE_SHA256 = {
    "transformers.models.t5.modeling_t5": "ee1a043f5c03c684a2bbb0213ea3b3f8be272a03ab884bdca57c31e7315e7b7e",
    "transformers.models.whisper.modeling_whisper": "851c78f1aa62f84ed0d0a4dfded478a05c98f43d26732b56fea2d4acc79728b8",
    "transformers.models.llama.modeling_llama": "31bf660a663259134324bc65da4e155951dc89c5ca46471d2325a9938e859e26",
    "transformers.models.mistral.modeling_mistral": "3fbb6202376a1fdf57166d199dd4d43f81226b1e0634ec8b61dcb1ade7de153e",
    "transformers.models.vit.modeling_vit": "069e0958c10e806551a2e1d2421cd26e6bff21f7c65aeb9f0f3c0dbde1a88ca5",
    "transformers.models.qwen2.modeling_qwen2": "a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb",
    "transformers.models.qwen3.modeling_qwen3": "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b",
    "transformers.models.phi.modeling_phi": "1853d5c78107af5d8bca539730451ecc060a91063a847b2c0acc2eb28e1cff4f",
    "transformers.models.gpt2.modeling_gpt2": "b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80",
    "transformers.pytorch_utils": "e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86",
    "transformers.modeling_outputs": "2edce49b0a7f9fad53b52395bece5fb20e3d0f83c305f3866354d19d790a5d14",
    "transformers.utils.generic": "94728825190dff491f2e0b93beb59c8246adb0b6049e3203c511405f43bb7e30",
    "transformers.activations": "3dd5a81b1e5e0c5371256f00efec19df52255420318737db1e03430a98cf4264",
}

_TORCH_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def native_module_outputs(module: nn.Module, input_spec: TensorSpec) -> dict[str, TensorSpec] | None:
    """Return only output ports established by a reviewed native implementation."""
    if not isinstance(input_spec, TensorSpec):
        return None
    family = type(module).__module__
    if family not in CONTRACT_SOURCE_SHA256 or not _trusted_source(type(module), family):
        return None
    if not _safe_module(module):
        return None

    if family == "transformers.models.t5.modeling_t5":
        from transformers.models.t5 import modeling_t5 as t5

        if type(module) is t5.T5Stack:
            return _t5_stack(module, input_spec, t5)
        if not _rank_three(input_spec):
            return None
        if type(module) is t5.T5Attention:
            width = _t5_attention(module, input_spec, t5)
            port = "0"
        elif type(module) in (t5.T5LayerSelfAttention, t5.T5LayerCrossAttention):
            width = _t5_attention_layer(module, input_spec, t5)
            port = "0"
        elif type(module) is t5.T5LayerFF:
            width = _t5_ff(module, input_spec, t5)
            port = "output"
        elif type(module) is t5.T5LayerNorm:
            candidate = input_spec.shape[-1]
            width = candidate if _t5_norm(module, candidate, _TORCH_DTYPES[input_spec.dtype], t5) else None
            port = "output"
        elif type(module) is t5.T5Block:
            width = _t5_block(module, input_spec, t5)
            port = "0"
        else:
            return None
    elif family == "transformers.activations":
        from transformers.activations import GELUActivation, SiLUActivation

        if not _rank_three(input_spec):
            return None
        if type(module) is GELUActivation:
            if module.act is not nn.functional.gelu:
                return None
        elif type(module) is not SiLUActivation:
            return None
        return {"output": TensorSpec(list(input_spec.shape), input_spec.dtype, "activation")}
    elif family == "transformers.models.llama.modeling_llama":
        from transformers.models.llama import modeling_llama as llama

        if not _rank_three(input_spec):
            return None
        if type(module) is llama.LlamaAttention:
            width = _llama_attention(module, input_spec, llama)
            port = "0"
        elif type(module) is llama.LlamaMLP:
            width = _llama_mlp(module, input_spec, llama)
            port = "output"
        elif type(module) is llama.LlamaRMSNorm:
            width = input_spec.shape[-1] if _llama_norm(module, input_spec, llama) else None
            port = "output"
        else:
            return None
    elif family == "transformers.models.mistral.modeling_mistral":
        from transformers.models.mistral import modeling_mistral as mistral

        if not _rank_three(input_spec):
            return None
        if type(module) is mistral.MistralAttention:
            width = _mistral_attention(module, input_spec, mistral)
            port = "0"
        elif type(module) is mistral.MistralMLP:
            width = _mistral_mlp(module, input_spec, mistral)
            port = "output"
        elif type(module) is mistral.MistralRMSNorm:
            width = input_spec.shape[-1] if _mistral_norm(module, input_spec, mistral) else None
            port = "output"
        elif type(module) is mistral.MistralDecoderLayer:
            width = _mistral_layer(module, input_spec, mistral)
            port = "output"
        else:
            return None
    elif family == "transformers.models.vit.modeling_vit":
        from transformers.models.vit import modeling_vit as vit

        if type(module) is not vit.ViTSelfAttention or not _rank_three(input_spec):
            return None
        output_width = _vit_attention(module, input_spec, vit)
        if output_width is None:
            return None
        return {"0": TensorSpec([*input_spec.shape[:2], output_width], input_spec.dtype, "hidden_states")}
    elif family in {
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
    }:
        if not _rank_three(input_spec):
            return None
        if family.endswith("qwen2.modeling_qwen2"):
            from transformers.models.qwen2 import modeling_qwen2 as qwen
            config_type, attention_type, mlp_type = qwen.Qwen2Config, qwen.Qwen2Attention, qwen.Qwen2MLP
            norm_type, layer_type = qwen.Qwen2RMSNorm, qwen.Qwen2DecoderLayer
            qwen3 = False
        else:
            from transformers.models.qwen3 import modeling_qwen3 as qwen
            config_type, attention_type, mlp_type = qwen.Qwen3Config, qwen.Qwen3Attention, qwen.Qwen3MLP
            norm_type, layer_type = qwen.Qwen3RMSNorm, qwen.Qwen3DecoderLayer
            qwen3 = True
        if type(module) is attention_type:
            width = _qwen_attention(module, input_spec, config_type, norm_type, qwen3)
        elif type(module) is mlp_type:
            width = _qwen_mlp(module, input_spec, config_type)
        elif type(module) is norm_type:
            width = input_spec.shape[-1] if _qwen_norm(module, input_spec, norm_type) else None
        elif type(module) is layer_type:
            width = _qwen_layer(module, input_spec, config_type, attention_type, mlp_type, norm_type, qwen3)
        else:
            return None
        port = "output" if type(module) in (norm_type, mlp_type, layer_type) else "0"
    elif family == "transformers.models.phi.modeling_phi":
        from transformers.models.phi import modeling_phi as phi

        if not _rank_three(input_spec):
            return None
        if type(module) is phi.PhiAttention:
            width = _phi_attention(module, input_spec, phi)
        elif type(module) is phi.PhiMLP:
            width = _phi_mlp(module, input_spec, phi)
        elif type(module) is phi.PhiDecoderLayer:
            width = _phi_layer(module, input_spec, phi)
        else:
            return None
        port = "0" if type(module) in (phi.PhiAttention, phi.PhiDecoderLayer) else "output"
    elif family == "transformers.models.gpt2.modeling_gpt2":
        from transformers.models.gpt2 import modeling_gpt2 as gpt2

        if type(module) is gpt2.GPT2Model:
            return _gpt2_model(module, input_spec, gpt2)
        if not _rank_three(input_spec):
            return None
        if type(module) is gpt2.GPT2Attention:
            width = _gpt2_attention(module, input_spec, gpt2)
        elif type(module) is gpt2.GPT2MLP:
            width = _gpt2_mlp(module, input_spec, gpt2)
        elif type(module) is gpt2.GPT2Block:
            width = _gpt2_block(module, input_spec, gpt2)
        else:
            return None
        port = "output" if type(module) is gpt2.GPT2MLP else "0"
    else:
        from transformers.models.whisper import modeling_whisper as whisper

        if not _rank_three(input_spec):
            return None
        if type(module) is whisper.WhisperAttention:
            width = _whisper_attention(module, input_spec, whisper)
        elif type(module) in (whisper.WhisperEncoderLayer, whisper.WhisperDecoderLayer):
            width = _whisper_layer(module, input_spec, whisper)
        else:
            return None
        port = "0"

    if width is None:
        return None
    return {port: TensorSpec(list(input_spec.shape), input_spec.dtype, "hidden_states")}


def _trusted_source(cls: type, family: str) -> bool:
    try:
        source = inspect.getsourcefile(cls)
        forward = inspect.unwrap(cls.forward)
        forward_source = inspect.getsourcefile(forward)
        return (
            bool(source)
            and forward_source == source
            and forward.__qualname__ == f"{cls.__name__}.forward"
            and hashlib.sha256(Path(source).read_bytes()).hexdigest() == CONTRACT_SOURCE_SHA256[family]
        )
    except (OSError, TypeError, KeyError):
        return False


def _trusted_module_file(module_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        source = getattr(module, "__file__", None)
        return bool(source) and hashlib.sha256(Path(source).read_bytes()).hexdigest() == CONTRACT_SOURCE_SHA256[module_name]
    except (ImportError, OSError, KeyError, TypeError):
        return False


def _rank_three(spec: TensorSpec) -> bool:
    return (
        isinstance(spec.shape, list)
        and len(spec.shape) == 3
        and all((type(axis) is int and axis > 0) or (type(axis) is str and bool(axis)) for axis in spec.shape)
        and type(spec.shape[-1]) is int
        and spec.dtype in _TORCH_DTYPES
    )


def _rank_two_ids(spec: TensorSpec) -> bool:
    return (
        isinstance(spec.shape, list)
        and len(spec.shape) == 2
        and all((type(axis) is int and axis > 0) or (type(axis) is str and bool(axis)) for axis in spec.shape)
        and spec.dtype == "I64"
    )


def _safe_module(module: nn.Module) -> bool:
    return (
        not module.training
        and "forward" not in vars(module)
        and not module._forward_hooks
        and not module._forward_pre_hooks
        and not module._backward_hooks
    )


def _linear(module: object, source: int, target: int, dtype: torch.dtype) -> bool:
    return (
        type(module) is nn.Linear
        and _safe_module(module)
        and module.in_features == source
        and module.out_features == target
        and tuple(module.weight.shape) == (target, source)
        and module.weight.dtype == dtype
        and (module.bias is None or (tuple(module.bias.shape) == (target,) and module.bias.dtype == dtype))
    )


def _t5_attention(module: nn.Module, spec: TensorSpec, t5: object) -> int | None:
    if not _safe_module(module) or not _trusted_source(type(module), "transformers.models.t5.modeling_t5"):
        return None
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    inner = getattr(module, "inner_dim", None)
    if (
        type(inner) is not int
        or inner <= 0
        or module.d_model != width
        or type(module.n_heads) is not int
        or type(module.key_value_proj_dim) is not int
        or module.n_heads * module.key_value_proj_dim != inner
        or module.pruned_heads
        or not all(_linear(getattr(module, name, None), width, inner, dtype) for name in ("q", "k", "v"))
        or not _linear(getattr(module, "o", None), inner, width, dtype)
    ):
        return None
    if module.has_relative_attention_bias:
        bias = getattr(module, "relative_attention_bias", None)
        if (
            type(bias) is not nn.Embedding
            or not _safe_module(bias)
            or bias.embedding_dim != module.n_heads
            or bias.weight.dtype != dtype
        ):
            return None
    return width


def _t5_norm(module: object, width: int, dtype: torch.dtype, t5: object) -> bool:
    return (
        type(module) is t5.T5LayerNorm
        and type(module).__module__ == "transformers.models.t5.modeling_t5"
        and _trusted_source(type(module), "transformers.models.t5.modeling_t5")
        and _safe_module(module)
        and tuple(module.weight.shape) == (width,)
        and module.weight.dtype == dtype
        and type(module.variance_epsilon) in (int, float)
        and math.isfinite(module.variance_epsilon)
        and module.variance_epsilon > 0
    )


def _dropout(module: object) -> bool:
    return type(module) is nn.Dropout and _safe_module(module)


def _t5_attention_layer(module: nn.Module, spec: TensorSpec, t5: object) -> int | None:
    if not _safe_module(module) or not _trusted_source(type(module), "transformers.models.t5.modeling_t5"):
        return None
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    child_name = "SelfAttention" if type(module) is t5.T5LayerSelfAttention else "EncDecAttention"
    attention = getattr(module, child_name, None)
    return (
        width
        if type(attention) is t5.T5Attention
        and _t5_attention(attention, spec, t5) == width
        and _t5_norm(getattr(module, "layer_norm", None), width, dtype, t5)
        and _dropout(getattr(module, "dropout", None))
        else None
    )


def _native_activation(module: object) -> bool:
    if not isinstance(module, nn.Module) or not _safe_module(module):
        return False
    from transformers import activations

    native_types = (activations.GELUActivation, activations.NewGELUActivation, activations.SiLUActivation)
    if type(module) is nn.ReLU:
        return True
    if type(module) is activations.GELUActivation and module.act is not nn.functional.gelu:
        return False
    return type(module) in native_types and _trusted_source(type(module), "transformers.activations")


def _t5_ff(module: nn.Module, spec: TensorSpec, t5: object) -> int | None:
    if not _safe_module(module) or not _trusted_source(type(module), "transformers.models.t5.modeling_t5"):
        return None
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    dense = getattr(module, "DenseReluDense", None)
    if (
        not _t5_norm(getattr(module, "layer_norm", None), width, dtype, t5)
        or not _dropout(getattr(module, "dropout", None))
        or type(dense) not in (t5.T5DenseActDense, t5.T5DenseGatedActDense)
        or not _safe_module(dense)
        or not _trusted_source(type(dense), "transformers.models.t5.modeling_t5")
        or not _native_activation(getattr(dense, "act", None))
    ):
        return None
    ff_width = getattr(getattr(dense, "wo", None), "in_features", None)
    if type(ff_width) is not int or ff_width <= 0 or not _linear(dense.wo, ff_width, width, dtype):
        return None
    first = ("wi",) if type(dense) is t5.T5DenseActDense else ("wi_0", "wi_1")
    return (
        width
        if all(_linear(getattr(dense, name, None), width, ff_width, dtype) for name in first)
        and _dropout(dense.dropout)
        else None
    )


def _t5_block(module: nn.Module, spec: TensorSpec, t5: object) -> int | None:
    layers = getattr(module, "layer", None)
    if type(layers) is not nn.ModuleList or not _safe_module(layers):
        return None
    expected = (
        (t5.T5LayerSelfAttention, t5.T5LayerCrossAttention, t5.T5LayerFF)
        if module.is_decoder
        else (t5.T5LayerSelfAttention, t5.T5LayerFF)
    )
    if len(layers) != len(expected) or any(type(layer) is not cls for layer, cls in zip(layers, expected)):
        return None
    for layer in layers:
        width = (
            _t5_ff(layer, spec, t5) if type(layer) is t5.T5LayerFF else _t5_attention_layer(layer, spec, t5)
        )
        if width is None:
            return None
    return spec.shape[-1]


def _t5_stack(module: nn.Module, ids: TensorSpec, t5: object) -> dict[str, TensorSpec] | None:
    if not _rank_two_ids(ids) or type(module.config) is not t5.T5Config:
        return None
    config = module.config
    width = getattr(config, "d_model", None)
    count = getattr(config, "num_layers", None)
    tokens = getattr(module, "embed_tokens", None)
    blocks = getattr(module, "block", None)
    if (
        type(width) is not int
        or width <= 0
        or type(count) is not int
        or count <= 0
        or type(module.is_decoder) is not bool
        or module.is_decoder is not config.is_decoder
        or module.model_parallel
        or type(tokens) is not nn.Embedding
        or not _safe_module(tokens)
        or tokens.embedding_dim != width
        or tuple(tokens.weight.shape) != (tokens.num_embeddings, width)
        or tokens.weight.dtype not in _TORCH_DTYPES.values()
        or type(blocks) is not nn.ModuleList
        or not _safe_module(blocks)
        or len(blocks) != count
        or not _t5_norm(getattr(module, "final_layer_norm", None), width, tokens.weight.dtype, t5)
        or not _dropout(getattr(module, "dropout", None))
    ):
        return None
    dtype = next(name for name, torch_dtype in _TORCH_DTYPES.items() if torch_dtype == tokens.weight.dtype)
    hidden = TensorSpec([*ids.shape, width], dtype, "hidden_states")
    for index, block in enumerate(blocks):
        if (
            type(block) is not t5.T5Block
            or block.is_decoder is not module.is_decoder
            or not _safe_module(block)
            or _t5_block(block, hidden, t5) != width
            or block.layer[0].SelfAttention.has_relative_attention_bias is not (index == 0)
        ):
            return None
    return {"0": TensorSpec([*ids.shape, width], dtype, "last_hidden_state")}


def _whisper_attention(module: nn.Module, spec: TensorSpec, whisper: object) -> int | None:
    if not _safe_module(module) or not _trusted_source(
        type(module), "transformers.models.whisper.modeling_whisper"
    ):
        return None
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    if (
        module.embed_dim != width
        or type(module.num_heads) is not int
        or type(module.head_dim) is not int
        or module.num_heads * module.head_dim != width
        or not all(
            _linear(getattr(module, name, None), width, width, dtype)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj")
        )
    ):
        return None
    return width


def _layer_norm(module: object, width: int, dtype: torch.dtype) -> bool:
    return (
        type(module) is nn.LayerNorm
        and _safe_module(module)
        and tuple(module.normalized_shape) == (width,)
        and module.weight is not None
        and tuple(module.weight.shape) == (width,)
        and module.weight.dtype == dtype
        and (module.bias is None or (tuple(module.bias.shape) == (width,) and module.bias.dtype == dtype))
    )


def _whisper_layer(module: nn.Module, spec: TensorSpec, whisper: object) -> int | None:
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    if (
        module.embed_dim != width
        or not _native_activation(getattr(module, "activation_fn", None))
        or not _layer_norm(getattr(module, "self_attn_layer_norm", None), width, dtype)
        or not _layer_norm(getattr(module, "final_layer_norm", None), width, dtype)
        or type(getattr(module, "self_attn", None)) is not whisper.WhisperAttention
        or _whisper_attention(module.self_attn, spec, whisper) is None
    ):
        return None
    ff_width = getattr(getattr(module, "fc1", None), "out_features", None)
    if (
        type(ff_width) is not int
        or ff_width <= 0
        or not _linear(module.fc1, width, ff_width, dtype)
        or not _linear(getattr(module, "fc2", None), ff_width, width, dtype)
    ):
        return None
    if type(module) is whisper.WhisperDecoderLayer and (
        not _layer_norm(getattr(module, "encoder_attn_layer_norm", None), width, dtype)
        or type(getattr(module, "encoder_attn", None)) is not whisper.WhisperAttention
        or _whisper_attention(module.encoder_attn, spec, whisper) is None
    ):
        return None
    return width


def _llama_attention(module: nn.Module, spec: TensorSpec, llama: object) -> int | None:
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    config = getattr(module, "config", None)
    if type(config) is not llama.LlamaConfig or config._attn_implementation != "eager":
        return None
    heads = getattr(config, "num_attention_heads", None)
    kv_heads = getattr(config, "num_key_value_heads", None)
    head_dim = getattr(module, "head_dim", None)
    if (
        config.hidden_size != width
        or type(heads) is not int
        or heads <= 0
        or type(kv_heads) is not int
        or kv_heads <= 0
        or heads % kv_heads != 0
        or type(head_dim) is not int
        or head_dim <= 0
        or module.num_key_value_groups != heads // kv_heads
        or module.scaling != head_dim**-0.5
    ):
        return None
    query_width, kv_width = heads * head_dim, kv_heads * head_dim
    return (
        width
        if _linear(getattr(module, "q_proj", None), width, query_width, dtype)
        and _linear(getattr(module, "k_proj", None), width, kv_width, dtype)
        and _linear(getattr(module, "v_proj", None), width, kv_width, dtype)
        and _linear(getattr(module, "o_proj", None), query_width, width, dtype)
        else None
    )


def _llama_mlp(module: nn.Module, spec: TensorSpec, llama: object) -> int | None:
    from transformers.activations import SiLUActivation

    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    config = getattr(module, "config", None)
    intermediate = getattr(module, "intermediate_size", None)
    activation = getattr(module, "act_fn", None)
    if (
        type(config) is not llama.LlamaConfig
        or config.hidden_size != width
        or config.intermediate_size != intermediate
        or config.hidden_act != "silu"
        or module.hidden_size != width
        or type(intermediate) is not int
        or intermediate <= 0
        or type(activation) is not SiLUActivation
        or not _safe_module(activation)
        or not _trusted_source(type(activation), "transformers.activations")
    ):
        return None
    return (
        width
        if _linear(getattr(module, "gate_proj", None), width, intermediate, dtype)
        and _linear(getattr(module, "up_proj", None), width, intermediate, dtype)
        and _linear(getattr(module, "down_proj", None), intermediate, width, dtype)
        else None
    )


def _llama_norm(module: nn.Module, spec: TensorSpec, llama: object) -> bool:
    return (
        type(module) is llama.LlamaRMSNorm
        and tuple(module.weight.shape) == (spec.shape[-1],)
        and module.weight.dtype == _TORCH_DTYPES[spec.dtype]
        and type(module.variance_epsilon) in (int, float)
        and math.isfinite(module.variance_epsilon)
        and module.variance_epsilon > 0
    )


def _mistral_attention(module: nn.Module, spec: TensorSpec, mistral: object) -> int | None:
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    config = getattr(module, "config", None)
    if type(config) is not mistral.MistralConfig or config._attn_implementation != "eager":
        return None
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = getattr(module, "head_dim", None)
    if (
        not _safe_module(module)
        or not _trusted_source(type(module), type(module).__module__)
        or config.hidden_size != width
        or type(heads) is not int or heads <= 0
        or type(kv_heads) is not int or kv_heads <= 0 or heads % kv_heads
        or type(head_dim) is not int or head_dim <= 0
        or module.num_key_value_groups != heads // kv_heads
        or module.scaling != head_dim ** -0.5
        or module.is_causal is not True
        or type(module.attention_dropout) not in (int, float)
        or not 0 <= module.attention_dropout <= 1
    ):
        return None
    query_width, kv_width = heads * head_dim, kv_heads * head_dim
    return width if (
        _linear(module.q_proj, width, query_width, dtype)
        and _linear(module.k_proj, width, kv_width, dtype)
        and _linear(module.v_proj, width, kv_width, dtype)
        and _linear(module.o_proj, query_width, width, dtype)
    ) else None


def _mistral_mlp(module: nn.Module, spec: TensorSpec, mistral: object) -> int | None:
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    config = getattr(module, "config", None)
    intermediate = getattr(module, "intermediate_size", None)
    if (
        not _safe_module(module)
        or not _trusted_source(type(module), type(module).__module__)
        or type(config) is not mistral.MistralConfig
        or config.hidden_size != width
        or config.intermediate_size != intermediate
        or module.hidden_size != width
        or type(intermediate) is not int or intermediate <= 0
        or not _native_activation(getattr(module, "act_fn", None))
    ):
        return None
    return width if (
        _linear(module.gate_proj, width, intermediate, dtype)
        and _linear(module.up_proj, width, intermediate, dtype)
        and _linear(module.down_proj, intermediate, width, dtype)
    ) else None


def _mistral_norm(module: nn.Module, spec: TensorSpec, mistral: object) -> bool:
    epsilon = getattr(module, "variance_epsilon", None)
    return (
        type(module) is mistral.MistralRMSNorm
        and _safe_module(module)
        and _trusted_source(type(module), type(module).__module__)
        and tuple(module.weight.shape) == (spec.shape[-1],)
        and module.weight.dtype == _TORCH_DTYPES[spec.dtype]
        and type(epsilon) in (int, float)
        and math.isfinite(epsilon)
        and epsilon > 0
    )


def _mistral_layer(module: nn.Module, spec: TensorSpec, mistral: object) -> int | None:
    width = spec.shape[-1]
    attention = getattr(module, "self_attn", None)
    mlp = getattr(module, "mlp", None)
    norms = (getattr(module, "input_layernorm", None), getattr(module, "post_attention_layernorm", None))
    return width if (
        type(attention) is mistral.MistralAttention
        and _mistral_attention(attention, spec, mistral) == width
        and type(mlp) is mistral.MistralMLP
        and _mistral_mlp(mlp, spec, mistral) == width
        and all(_mistral_norm(norm, spec, mistral) for norm in norms)
    ) else None


def _vit_attention(module: nn.Module, spec: TensorSpec, vit: object) -> int | None:
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    config = getattr(module, "config", None)
    if type(config) is not vit.ViTConfig or config._attn_implementation != "eager":
        return None
    heads = getattr(module, "num_attention_heads", None)
    head_size = getattr(module, "attention_head_size", None)
    output_width = getattr(module, "all_head_size", None)
    if (
        config.hidden_size != width
        or type(heads) is not int
        or heads <= 0
        or heads != config.num_attention_heads
        or type(head_size) is not int
        or head_size <= 0
        or head_size != int(width / heads)
        or type(output_width) is not int
        or output_width != heads * head_size
        or module.scaling != head_size**-0.5
        or module.is_causal is not False
    ):
        return None
    return (
        output_width
        if all(
            _linear(getattr(module, name, None), width, output_width, dtype)
            for name in ("query", "key", "value")
        )
        else None
    )


def _qwen_norm(module: object, spec: TensorSpec, norm_type: type) -> bool:
    epsilon = getattr(module, "variance_epsilon", None)
    return (
        type(module) is norm_type
        and _safe_module(module)
        and _trusted_source(type(module), type(module).__module__)
        and tuple(module.weight.shape) == (spec.shape[-1],)
        and module.weight.dtype == _TORCH_DTYPES[spec.dtype]
        and type(epsilon) in (int, float)
        and math.isfinite(epsilon)
        and epsilon > 0
    )


def _qwen_attention(module: nn.Module, spec: TensorSpec, config_type: type, norm_type: type, qwen3: bool) -> int | None:
    config = getattr(module, "config", None)
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    heads = getattr(config, "num_attention_heads", None)
    kv_heads = getattr(config, "num_key_value_heads", None)
    head_dim = getattr(module, "head_dim", None)
    if (
        not _safe_module(module)
        or not _trusted_source(type(module), type(module).__module__)
        or type(config) is not config_type
        or config._attn_implementation != "eager"
        or config.hidden_size != width
        or type(heads) is not int or heads <= 0
        or type(kv_heads) is not int or kv_heads <= 0 or heads % kv_heads
        or type(head_dim) is not int or head_dim <= 0
        or module.num_key_value_groups != heads // kv_heads
        or module.scaling != head_dim ** -0.5
        or module.is_causal is not True
    ):
        return None
    query_width, kv_width = heads * head_dim, kv_heads * head_dim
    if not all((
        _linear(module.q_proj, width, query_width, dtype),
        _linear(module.k_proj, width, kv_width, dtype),
        _linear(module.v_proj, width, kv_width, dtype),
        _linear(module.o_proj, query_width, width, dtype),
    )):
        return None
    if qwen3:
        head_spec = TensorSpec(list(spec.shape[:-1]) + [head_dim], spec.dtype, "head_states")
        if not _qwen_norm(module.q_norm, head_spec, norm_type):
            return None
        if not _qwen_norm(module.k_norm, head_spec, norm_type):
            return None
    return width


def _qwen_mlp(module: nn.Module, spec: TensorSpec, config_type: type) -> int | None:
    config = getattr(module, "config", None)
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    intermediate = getattr(config, "intermediate_size", None)
    if (
        not _safe_module(module)
        or not _trusted_source(type(module), type(module).__module__)
        or type(config) is not config_type
        or config._attn_implementation != "eager"
        or config.hidden_size != width
        or type(intermediate) is not int or intermediate <= 0
        or module.hidden_size != width
        or module.intermediate_size != intermediate
        or not _native_activation(module.act_fn)
    ):
        return None
    return width if all(
        _linear(getattr(module, name, None), width, intermediate, dtype)
        for name in ("gate_proj", "up_proj")
    ) and _linear(module.down_proj, intermediate, width, dtype) else None


def _qwen_layer(module: nn.Module, spec: TensorSpec, config_type: type, attention_type: type,
                mlp_type: type, norm_type: type, qwen3: bool) -> int | None:
    if not _safe_module(module) or not _trusted_source(type(module), type(module).__module__):
        return None
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    attention, mlp = getattr(module, "self_attn", None), getattr(module, "mlp", None)
    norms = (getattr(module, "input_layernorm", None), getattr(module, "post_attention_layernorm", None))
    config = getattr(attention, "config", None)
    return width if (
        type(config) is config_type and config.hidden_size == width
        and type(attention) is attention_type and _qwen_attention(attention, spec, config_type, norm_type, qwen3) == width
        and type(mlp) is mlp_type and _qwen_mlp(mlp, spec, config_type) == width
        and all(_qwen_norm(norm, spec, norm_type) for norm in norms)
        and all(norm.weight.dtype == dtype for norm in norms)
    ) else None


def _phi_attention(module: nn.Module, spec: TensorSpec, phi: object) -> int | None:
    config = getattr(module, "config", None)
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    heads, kv_heads = getattr(config, "num_attention_heads", None), getattr(config, "num_key_value_heads", None)
    head_dim = getattr(module, "head_dim", None)
    if (not _safe_module(module) or not _trusted_source(type(module), type(module).__module__)
        or type(config) is not phi.PhiConfig or config._attn_implementation != "eager"
        or config.hidden_size != width or type(heads) is not int or heads <= 0
        or type(kv_heads) is not int or kv_heads <= 0 or heads % kv_heads
        or type(head_dim) is not int or head_dim <= 0 or module.is_causal is not True
        or module.scaling != head_dim ** -0.5 or module.num_key_value_groups != heads // kv_heads):
        return None
    query_width, kv_width = heads * head_dim, kv_heads * head_dim
    if not (_linear(module.q_proj, width, query_width, dtype)
        and _linear(module.k_proj, width, kv_width, dtype)
        and _linear(module.v_proj, width, kv_width, dtype)
        and _linear(module.dense, query_width, width, dtype)):
        return None
    if config.qk_layernorm and not all(
        _layer_norm(getattr(module, name, None), head_dim, dtype) for name in ("q_layernorm", "k_layernorm")
    ):
        return None
    return width


def _phi_mlp(module: nn.Module, spec: TensorSpec, phi: object) -> int | None:
    config = getattr(module, "config", None)
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    intermediate = getattr(config, "intermediate_size", None)
    if (not _safe_module(module) or not _trusted_source(type(module), type(module).__module__)
        or type(config) is not phi.PhiConfig or config._attn_implementation != "eager"
        or config.hidden_size != width or type(intermediate) is not int or intermediate <= 0
        or not _native_activation(module.activation_fn)):
        return None
    return width if _linear(module.fc1, width, intermediate, dtype) and _linear(module.fc2, intermediate, width, dtype) else None


def _phi_layer(module: nn.Module, spec: TensorSpec, phi: object) -> int | None:
    if not _safe_module(module) or not _trusted_source(type(module), type(module).__module__):
        return None
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    attention, mlp = getattr(module, "self_attn", None), getattr(module, "mlp", None)
    return width if (
        type(attention) is phi.PhiAttention and _phi_attention(attention, spec, phi) == width
        and type(mlp) is phi.PhiMLP and _phi_mlp(mlp, spec, phi) == width
        and _layer_norm(module.input_layernorm, width, dtype)
        and _dropout(module.resid_dropout)
    ) else None


def _gpt2_conv1d(module: object, source: int, target: int, dtype: torch.dtype, gpt2: object) -> bool:
    return (
        type(module) is gpt2.Conv1D and _safe_module(module)
        and _trusted_source(type(module), "transformers.pytorch_utils")
        and module.nf == target
        and tuple(module.weight.shape) == (source, target) and module.weight.dtype == dtype
        and tuple(module.bias.shape) == (target,) and module.bias.dtype == dtype
    )


def _gpt2_attention(module: nn.Module, spec: TensorSpec, gpt2: object) -> int | None:
    config = getattr(module, "config", None)
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    heads = getattr(module, "num_heads", None)
    if (not _safe_module(module) or not _trusted_source(type(module), type(module).__module__)
        or type(config) is not gpt2.GPT2Config or config._attn_implementation != "eager"
        or config.add_cross_attention or module.is_cross_attention or config.hidden_size != width
        or type(heads) is not int or heads <= 0 or heads != config.num_attention_heads
        or module.head_dim * heads != width or module.split_size != width or module.is_causal is not True
        or module.pruned_heads):
        return None
    return width if (_gpt2_conv1d(module.c_attn, width, 3 * width, dtype, gpt2)
        and _gpt2_conv1d(module.c_proj, width, width, dtype, gpt2)
        and _dropout(module.attn_dropout) and _dropout(module.resid_dropout)) else None


def _gpt2_mlp(module: nn.Module, spec: TensorSpec, gpt2: object) -> int | None:
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    intermediate = getattr(module.c_fc, "nf", None)
    if (not _safe_module(module) or not _trusted_source(type(module), type(module).__module__)
        or type(intermediate) is not int or intermediate <= 0
        or not _native_activation(module.act)):
        return None
    return width if (_gpt2_conv1d(module.c_fc, width, intermediate, dtype, gpt2)
        and _gpt2_conv1d(module.c_proj, intermediate, width, dtype, gpt2)
        and _dropout(module.dropout)) else None


def _gpt2_block(module: nn.Module, spec: TensorSpec, gpt2: object) -> int | None:
    if not _safe_module(module) or not _trusted_source(type(module), type(module).__module__):
        return None
    config = getattr(getattr(module, "attn", None), "config", None)
    width, dtype = spec.shape[-1], _TORCH_DTYPES[spec.dtype]
    return width if (
        type(config) is gpt2.GPT2Config and not config.add_cross_attention and config.hidden_size == width
        and type(module.attn) is gpt2.GPT2Attention and _gpt2_attention(module.attn, spec, gpt2) == width
        and type(module.mlp) is gpt2.GPT2MLP and _gpt2_mlp(module.mlp, spec, gpt2) == width
        and _layer_norm(module.ln_1, width, dtype) and _layer_norm(module.ln_2, width, dtype)
    ) else None


def _trusted_class_method(cls: type, method_name: str, module_name: str) -> bool:
    try:
        method = inspect.unwrap(getattr(cls, method_name))
        source = inspect.getsourcefile(method)
        module = importlib.import_module(module_name)
        return (
            bool(source)
            and source == getattr(module, "__file__", None)
            and method.__qualname__ == f"{cls.__name__}.{method_name}"
        )
    except (AttributeError, ImportError, OSError, TypeError):
        return False


def _gpt2_model_output_zero_is_hidden_state(gpt2: object) -> bool:
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    from transformers.utils.generic import ModelOutput

    output_type = BaseModelOutputWithPastAndCrossAttentions
    return (
        output_type.__module__ == "transformers.modeling_outputs"
        and getattr(gpt2, "BaseModelOutputWithPastAndCrossAttentions", None) is output_type
        and output_type.__bases__ == (ModelOutput,)
        and tuple(output_type.__dataclass_fields__) == (
            "last_hidden_state",
            "past_key_values",
            "hidden_states",
            "attentions",
            "cross_attentions",
        )
        and _trusted_module_file("transformers.modeling_outputs")
        and _trusted_module_file("transformers.utils.generic")
        and all(
            _trusted_class_method(ModelOutput, method_name, "transformers.utils.generic")
            for method_name in ("__getitem__", "to_tuple", "__post_init__")
        )
    )


def _gpt2_model(module: nn.Module, ids: TensorSpec, gpt2: object) -> dict[str, TensorSpec] | None:
    config = getattr(module, "config", None)
    width = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    position_count = getattr(config, "max_position_embeddings", None)
    layer_count = getattr(config, "n_layer", None)
    tokens, positions = getattr(module, "wte", None), getattr(module, "wpe", None)
    layers, final_norm = getattr(module, "h", None), getattr(module, "ln_f", None)
    if (
        not _rank_two_ids(ids)
        or type(config) is not gpt2.GPT2Config
        or config._attn_implementation != "eager"
        or config.add_cross_attention
        or type(width) is not int or width <= 0
        or type(vocab_size) is not int or vocab_size <= 0
        or type(position_count) is not int or position_count <= 0
        or type(layer_count) is not int or layer_count <= 0
        or module.model_parallel
        or getattr(module, "gradient_checkpointing", False)
        or type(tokens) is not nn.Embedding
        or not _safe_module(tokens)
        or tokens.num_embeddings != vocab_size
        or tokens.embedding_dim != width
        or tuple(tokens.weight.shape) != (vocab_size, width)
        or tokens.weight.dtype not in _TORCH_DTYPES.values()
        or type(positions) is not nn.Embedding
        or not _safe_module(positions)
        or positions.num_embeddings != position_count
        or positions.embedding_dim != width
        or tuple(positions.weight.shape) != (position_count, width)
        or positions.weight.dtype != tokens.weight.dtype
        or type(layers) is not nn.ModuleList
        or not _safe_module(layers)
        or len(layers) != layer_count
        or any(type(block) is not gpt2.GPT2Block or _gpt2_block(
            block, TensorSpec([*ids.shape, width], _dtype_name(tokens.weight.dtype), "hidden_states"), gpt2
        ) != width for block in layers)
        or not _layer_norm(final_norm, width, tokens.weight.dtype)
        or final_norm.eps != config.layer_norm_epsilon
        or not _dropout(getattr(module, "drop", None))
        or not _gpt2_model_output_zero_is_hidden_state(gpt2)
    ):
        return None
    return {"0": TensorSpec([*ids.shape, width], _dtype_name(tokens.weight.dtype), "last_hidden_state")}


def _dtype_name(dtype: torch.dtype) -> str | None:
    return next((name for name, candidate in _TORCH_DTYPES.items() if candidate == dtype), None)
