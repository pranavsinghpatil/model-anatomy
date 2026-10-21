"""Reviewed native contracts inspect meta modules without running forwards."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.activations import GELUActivation, SiLUActivation
from transformers.models.llama import modeling_llama as llama
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.t5 import modeling_t5 as t5
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.vit import modeling_vit as vit
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.whisper import modeling_whisper as whisper
from transformers.models.whisper.configuration_whisper import WhisperConfig

from model_anatomy.src.ir import TensorSpec
from model_anatomy.src.native_output_contracts import CONTRACT_SOURCE_SHA256, native_module_outputs


def _t5_config(decoder: bool = False) -> T5Config:
    return T5Config(d_model=16, d_kv=8, d_ff=32, num_heads=2, is_decoder=decoder)


def _whisper_config() -> WhisperConfig:
    return WhisperConfig(
        d_model=16,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
    )


def _llama_config() -> LlamaConfig:
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1)
    config._attn_implementation = "eager"
    return config


def _vit_config() -> ViTConfig:
    config = ViTConfig(hidden_size=16, num_attention_heads=2)
    config._attn_implementation = "eager"
    return config


def _module(kind: str) -> nn.Module:
    with torch.device("meta"):
        if kind == "t5_attention":
            module = t5.T5Attention(_t5_config(), layer_idx=0)
        elif kind == "t5_self":
            module = t5.T5LayerSelfAttention(_t5_config(), layer_idx=0)
        elif kind == "t5_cross":
            module = t5.T5LayerCrossAttention(_t5_config(True), layer_idx=0)
        elif kind == "t5_ff":
            module = t5.T5LayerFF(_t5_config())
        elif kind == "t5_norm":
            module = t5.T5LayerNorm(16, eps=1e-6)
        elif kind == "t5_block":
            module = t5.T5Block(_t5_config(), layer_idx=0)
        elif kind == "t5_decoder_block":
            module = t5.T5Block(_t5_config(True), layer_idx=0)
        elif kind == "t5_stack":
            config = _t5_config()
            config.num_layers = 2
            module = t5.T5Stack(config, nn.Embedding(32, 16))
        elif kind == "t5_decoder_stack":
            config = _t5_config(True)
            config.num_layers = 2
            module = t5.T5Stack(config, nn.Embedding(32, 16))
        elif kind == "whisper_attention":
            module = whisper.WhisperAttention(16, 2, config=_whisper_config())
        elif kind == "whisper_encoder":
            module = whisper.WhisperEncoderLayer(_whisper_config())
        elif kind == "whisper_decoder":
            module = whisper.WhisperDecoderLayer(_whisper_config(), layer_idx=0)
        elif kind == "llama_attention":
            module = llama.LlamaAttention(_llama_config(), layer_idx=0)
        elif kind == "llama_mlp":
            module = llama.LlamaMLP(_llama_config())
        elif kind == "llama_norm":
            module = llama.LlamaRMSNorm(16, eps=1e-6)
        elif kind == "vit_attention":
            module = vit.ViTSelfAttention(_vit_config())
        else:
            raise AssertionError(kind)
    return module.eval()


SPEC = TensorSpec(["B", "S", 16], "F32", "hidden_states")


@pytest.mark.parametrize(
    ("kind", "port"),
    [
        ("t5_attention", "0"),
        ("t5_self", "0"),
        ("t5_cross", "0"),
        ("t5_ff", "output"),
        ("t5_norm", "output"),
        ("t5_block", "0"),
        ("t5_decoder_block", "0"),
        ("whisper_attention", "0"),
        ("whisper_encoder", "0"),
        ("whisper_decoder", "0"),
        ("llama_attention", "0"),
        ("llama_mlp", "output"),
        ("llama_norm", "output"),
        ("vit_attention", "0"),
    ],
)
def test_validated_native_module_claims_only_its_primary_port(kind: str, port: str) -> None:
    outputs = native_module_outputs(_module(kind), SPEC)
    assert outputs == {port: TensorSpec(["B", "S", 16], "F32", "hidden_states")}


@pytest.mark.parametrize("kind", ["t5_stack", "t5_decoder_stack"])
def test_t5_stack_ids_contract_uses_actual_embedding_and_block_metadata(kind: str) -> None:
    module = _module(kind)
    ids = TensorSpec(["B", "S"], "I64", "input_ids")
    assert native_module_outputs(module, ids) == {"0": TensorSpec(["B", "S", 16], "F32", "last_hidden_state")}
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize("replacement", ["embed_tokens", "final_layer_norm", "dropout"])
def test_t5_stack_replaced_terminal_child_rejects_contract(replacement: str) -> None:
    module = _module("t5_stack")
    setattr(module, replacement, nn.Identity())
    assert native_module_outputs(module, TensorSpec(["B", "S"], "I64", "input_ids")) is None


def test_t5_stack_replaced_block_or_count_rejects_contract() -> None:
    module = _module("t5_stack")
    ids = TensorSpec(["B", "S"], "I64", "input_ids")
    module.block[0] = nn.Identity()
    assert native_module_outputs(module, ids) is None
    module = _module("t5_stack")
    module.config.num_layers += 1
    assert native_module_outputs(module, ids) is None


def test_t5_stack_requires_integer_token_ids_and_matching_meta_dtype() -> None:
    module = _module("t5_stack")
    assert native_module_outputs(module, TensorSpec(["B", "S"], "F32", "input_ids")) is None
    module.embed_tokens.half()
    assert native_module_outputs(module, TensorSpec(["B", "S"], "I64", "input_ids")) is None


def test_reviewed_gelu_preserves_known_shape_and_dtype_only_with_native_act() -> None:
    module = GELUActivation().eval()
    assert native_module_outputs(module, SPEC) == {"output": TensorSpec(["B", "S", 16], "F32", "activation")}
    assert native_module_outputs(GELUActivation(use_gelu_python=True).eval(), SPEC) is None
    module.act = lambda value: value
    assert native_module_outputs(module, SPEC) is None


def test_reviewed_silu_preserves_shape_and_dtype() -> None:
    module = SiLUActivation().eval()
    assert native_module_outputs(module, SPEC) == {"output": TensorSpec(["B", "S", 16], "F32", "activation")}
    module.forward = lambda value: value
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize(
    "kind", ["t5_block", "whisper_encoder", "llama_attention", "llama_mlp", "vit_attention"]
)
def test_fingerprint_mismatch_rejects_contract(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    module = _module(kind)
    family = type(module).__module__
    monkeypatch.setitem(CONTRACT_SOURCE_SHA256, family, "0" * 64)
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize(
    ("kind", "replace"),
    [
        ("t5_attention", lambda m: setattr(m, "q", nn.Identity())),
        ("t5_self", lambda m: setattr(m, "SelfAttention", nn.Identity())),
        ("t5_cross", lambda m: setattr(m, "layer_norm", nn.Identity())),
        ("t5_ff", lambda m: setattr(m, "DenseReluDense", nn.Identity())),
        ("t5_block", lambda m: m.layer.__setitem__(0, nn.Identity())),
        ("whisper_attention", lambda m: setattr(m, "out_proj", nn.Identity())),
        ("whisper_encoder", lambda m: setattr(m, "fc2", nn.Identity())),
        ("whisper_decoder", lambda m: setattr(m, "encoder_attn", nn.Identity())),
        ("llama_attention", lambda m: setattr(m, "o_proj", nn.Identity())),
        ("llama_mlp", lambda m: setattr(m, "act_fn", nn.Identity())),
        ("vit_attention", lambda m: setattr(m, "query", nn.Identity())),
    ],
)
def test_replaced_child_rejects_contract(kind: str, replace) -> None:
    module = _module(kind)
    replace(module)
    assert native_module_outputs(module, SPEC) is None


def test_subclass_is_not_treated_as_reviewed_native_class() -> None:
    class CustomT5Attention(t5.T5Attention):
        pass

    with torch.device("meta"):
        module = CustomT5Attention(_t5_config(), layer_idx=0)
    assert native_module_outputs(module.eval(), SPEC) is None


@pytest.mark.parametrize(
    "spec",
    [
        TensorSpec(None, "F32", "hidden_states"),
        TensorSpec(["B", "S"], "F32", "hidden_states"),
        TensorSpec(["B", "S", "H"], "F32", "hidden_states"),
        TensorSpec(["B", "S", 8], "F32", "hidden_states"),
        TensorSpec(["B", "S", 16], None, "hidden_states"),
        TensorSpec(["B", "S", 16], "F16", "hidden_states"),
    ],
)
def test_unproved_input_or_weight_dtype_is_rejected(spec: TensorSpec) -> None:
    assert native_module_outputs(_module("t5_attention"), spec) is None


@pytest.mark.parametrize("kind", ["t5_block", "whisper_encoder", "whisper_decoder"])
def test_half_precision_contract_follows_actual_meta_weight_dtype(kind: str) -> None:
    module = _module(kind).half()
    spec = TensorSpec(["B", "S", 16], "F16", "hidden_states")
    assert native_module_outputs(module, spec) == {"0": TensorSpec(["B", "S", 16], "F16", "hidden_states")}
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize("dtype_name,conversion", [("F16", "half"), ("BF16", "bfloat16")])
def test_t5_norm_half_precision_cast_back_matches_native_weight(dtype_name: str, conversion: str) -> None:
    module = getattr(_module("t5_norm"), conversion)()
    spec = TensorSpec(["B", "S", 16], dtype_name, "hidden_states")
    assert native_module_outputs(module, spec) == {
        "output": TensorSpec(["B", "S", 16], dtype_name, "hidden_states")
    }


@pytest.mark.parametrize("epsilon", [0, -1.0, float("nan"), "invalid"])
def test_t5_norm_unvalidated_epsilon_is_rejected(epsilon: object) -> None:
    module = _module("t5_norm")
    module.variance_epsilon = epsilon
    assert native_module_outputs(module, SPEC) is None


def test_train_mode_or_forward_override_is_rejected() -> None:
    module = _module("whisper_attention")
    module.train()
    assert native_module_outputs(module, SPEC) is None
    module.eval()
    module.forward = lambda *_args, **_kwargs: None
    assert native_module_outputs(module, SPEC) is None


def test_nested_forward_override_is_rejected() -> None:
    module = _module("t5_block")
    module.layer[0].forward = lambda *_args, **_kwargs: None
    assert native_module_outputs(module, SPEC) is None


def test_replaced_native_class_forward_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module("t5_attention")
    monkeypatch.setattr(t5.T5Attention, "forward", lambda *_args, **_kwargs: None)
    assert native_module_outputs(module, SPEC) is None


def test_llama_attention_requires_eager_and_matching_projection_geometry() -> None:
    module = _module("llama_attention")
    module.config._attn_implementation = "sdpa"
    assert native_module_outputs(module, SPEC) is None
    module.config._attn_implementation = "eager"
    module.num_key_value_groups = 1
    assert native_module_outputs(module, SPEC) is None
    module = _module("llama_attention")
    with torch.device("meta"):
        module.k_proj = nn.Linear(16, 16)
    assert native_module_outputs(module, SPEC) is None


def test_llama_mlp_rejects_mismatched_width_dtype_or_activation() -> None:
    module = _module("llama_mlp")
    assert native_module_outputs(module, TensorSpec(["B", "S", 8], "F32", "hidden_states")) is None
    assert native_module_outputs(module, TensorSpec(["B", "S", 16], "F16", "hidden_states")) is None
    module.config.hidden_act = "gelu"
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize("epsilon", [0, -1.0, float("nan"), "invalid"])
def test_llama_norm_rejects_invalid_epsilon(epsilon: object) -> None:
    module = _module("llama_norm")
    module.variance_epsilon = epsilon
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize(
    "kind,port", [("llama_attention", "0"), ("llama_mlp", "output"), ("llama_norm", "output")]
)
def test_llama_half_precision_uses_actual_meta_weight_dtype(kind: str, port: str) -> None:
    module = _module(kind).half()
    spec = TensorSpec(["B", "S", 16], "F16", "hidden_states")
    assert native_module_outputs(module, spec) == {port: TensorSpec(["B", "S", 16], "F16", "hidden_states")}
    assert native_module_outputs(module, SPEC) is None


def test_vit_attention_requires_eager_and_matching_projection_geometry() -> None:
    module = _module("vit_attention")
    module.config._attn_implementation = "sdpa"
    assert native_module_outputs(module, SPEC) is None
    module.config._attn_implementation = "eager"
    module.all_head_size = 8
    assert native_module_outputs(module, SPEC) is None
    module = _module("vit_attention")
    with torch.device("meta"):
        module.value = nn.Linear(16, 8)
    assert native_module_outputs(module, SPEC) is None


def test_vit_attention_output_uses_actual_all_head_size() -> None:
    config = _vit_config()
    config.num_attention_heads = 3
    config.embedding_size = 16
    with torch.device("meta"):
        module = vit.ViTSelfAttention(config).eval()
    assert native_module_outputs(module, SPEC) == {"0": TensorSpec(["B", "S", 15], "F32", "hidden_states")}
