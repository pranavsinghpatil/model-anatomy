"""Narrow output contracts for reviewed native T5, Whisper, Llama, and ViT modules.

These describe the canonical eval, no-cache, no-autocast profile. They inspect
module structure and parameter metadata only; no forward method is called.
"""

from __future__ import annotations

import hashlib
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
    "transformers.models.vit.modeling_vit": "069e0958c10e806551a2e1d2421cd26e6bff21f7c65aeb9f0f3c0dbde1a88ca5",
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
    elif family == "transformers.models.vit.modeling_vit":
        from transformers.models.vit import modeling_vit as vit

        if type(module) is not vit.ViTSelfAttention or not _rank_three(input_spec):
            return None
        output_width = _vit_attention(module, input_spec, vit)
        if output_width is None:
            return None
        return {"0": TensorSpec([*input_spec.shape[:2], output_width], input_spec.dtype, "hidden_states")}
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
        and module.weight.dtype == dtype
        and (module.bias is None or module.bias.dtype == dtype)
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
