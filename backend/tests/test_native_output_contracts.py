"""Reviewed native contracts inspect meta modules without running forwards."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch
from torch import nn
from transformers.activations import GELUActivation, SiLUActivation
from transformers.models.gpt2 import modeling_gpt2 as gpt2
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.llama import modeling_llama as llama
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.phi import modeling_phi as phi
from transformers.models.phi.configuration_phi import PhiConfig
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3 import modeling_qwen3 as qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
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


def _qwen_config(kind: str):
    config_type = Qwen2Config if kind == "qwen2" else Qwen3Config
    config = config_type(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        **({"head_dim": 8} if kind == "qwen3" else {}),
    )
    config._attn_implementation = "eager"
    return config


def _phi_config() -> PhiConfig:
    config = PhiConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2)
    config._attn_implementation = "eager"
    return config


def _gpt2_config() -> GPT2Config:
    config = GPT2Config(n_embd=16, n_head=2, n_layer=1)
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
        elif kind == "qwen2_attention":
            module = qwen2.Qwen2Attention(_qwen_config("qwen2"), layer_idx=0)
        elif kind == "qwen2_mlp":
            module = qwen2.Qwen2MLP(_qwen_config("qwen2"))
        elif kind == "qwen2_norm":
            module = qwen2.Qwen2RMSNorm(16)
        elif kind == "qwen2_layer":
            module = qwen2.Qwen2DecoderLayer(_qwen_config("qwen2"), layer_idx=0)
        elif kind == "qwen3_attention":
            module = qwen3.Qwen3Attention(_qwen_config("qwen3"), layer_idx=0)
        elif kind == "qwen3_mlp":
            module = qwen3.Qwen3MLP(_qwen_config("qwen3"))
        elif kind == "qwen3_norm":
            module = qwen3.Qwen3RMSNorm(16)
        elif kind == "qwen3_layer":
            module = qwen3.Qwen3DecoderLayer(_qwen_config("qwen3"), layer_idx=0)
        elif kind == "phi_attention":
            module = phi.PhiAttention(_phi_config(), layer_idx=0)
        elif kind == "phi_mlp":
            module = phi.PhiMLP(_phi_config())
        elif kind == "phi_layer":
            module = phi.PhiDecoderLayer(_phi_config(), layer_idx=0)
        elif kind == "gpt2_attention":
            module = gpt2.GPT2Attention(_gpt2_config(), layer_idx=0)
        elif kind == "gpt2_mlp":
            module = gpt2.GPT2MLP(32, _gpt2_config())
        elif kind == "gpt2_block":
            module = gpt2.GPT2Block(_gpt2_config(), layer_idx=0)
        elif kind == "gpt2_model":
            module = gpt2.GPT2Model(_gpt2_config())
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
        ("qwen2_attention", "0"),
        ("qwen2_mlp", "output"),
        ("qwen2_norm", "output"),
        ("qwen2_layer", "output"),
        ("qwen3_attention", "0"),
        ("qwen3_mlp", "output"),
        ("qwen3_norm", "output"),
        ("qwen3_layer", "output"),
        ("phi_attention", "0"),
        ("phi_mlp", "output"),
        ("phi_layer", "0"),
        ("gpt2_attention", "0"),
        ("gpt2_mlp", "output"),
        ("gpt2_block", "0"),
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


def test_gpt2_model_contract_maps_only_proven_first_hidden_state_port() -> None:
    module = _module("gpt2_model")
    ids = TensorSpec(["B", "S"], "I64", "input_ids")
    assert native_module_outputs(module, ids) == {"0": TensorSpec(["B", "S", 16], "F32", "last_hidden_state")}
    assert native_module_outputs(module, SPEC) is None
    assert native_module_outputs(module, TensorSpec(["B", "S"], "F32", "input_ids")) is None
    assert native_module_outputs(module, TensorSpec(["B", "S", 1], "I64", "input_ids")) is None


def test_gpt2_model_contract_derives_hidden_dtype_from_embeddings() -> None:
    module = _module("gpt2_model").half()
    ids = TensorSpec(["B", "S"], "I64", "input_ids")
    assert native_module_outputs(module, ids) == {"0": TensorSpec(["B", "S", 16], "F16", "last_hidden_state")}
    module.wpe.float()
    assert native_module_outputs(module, ids) is None


@pytest.mark.parametrize("replacement", ["wte", "wpe", "drop", "ln_f"])
def test_gpt2_model_contract_rejects_replaced_terminal_child(replacement: str) -> None:
    module = _module("gpt2_model")
    setattr(module, replacement, nn.Identity())
    assert native_module_outputs(module, TensorSpec(["B", "S"], "I64", "input_ids")) is None


def test_gpt2_model_contract_rejects_bad_block_type_count_and_mode() -> None:
    ids = TensorSpec(["B", "S"], "I64", "input_ids")
    module = _module("gpt2_model")
    module.h[0] = nn.Identity()
    assert native_module_outputs(module, ids) is None
    module = _module("gpt2_model")
    module.config.n_layer += 1
    assert native_module_outputs(module, ids) is None
    module = _module("gpt2_model")
    module.train()
    assert native_module_outputs(module, ids) is None


def test_gpt2_model_contract_rejects_bad_output_wrapper_source(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module("gpt2_model")
    monkeypatch.setitem(CONTRACT_SOURCE_SHA256, "transformers.modeling_outputs", "0" * 64)
    assert native_module_outputs(module, TensorSpec(["B", "S"], "I64", "input_ids")) is None


def test_gpt2_model_contract_rejects_rebound_output_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module("gpt2_model")
    monkeypatch.setattr(gpt2, "BaseModelOutputWithPastAndCrossAttentions", object())
    assert native_module_outputs(module, TensorSpec(["B", "S"], "I64", "input_ids")) is None


@pytest.mark.parametrize("method_name", ["__getitem__", "to_tuple", "__post_init__"])
def test_gpt2_model_contract_rejects_replaced_output_wrapper_method(
    monkeypatch: pytest.MonkeyPatch, method_name: str
) -> None:
    from transformers.utils.generic import ModelOutput

    module = _module("gpt2_model")
    monkeypatch.setattr(ModelOutput, method_name, lambda *_args, **_kwargs: None)
    assert native_module_outputs(module, TensorSpec(["B", "S"], "I64", "input_ids")) is None


def test_gpt2_model_output_wrapper_field_order_proves_index_zero_is_hidden_state() -> None:
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    from transformers.utils.generic import ModelOutput

    output_type = BaseModelOutputWithPastAndCrossAttentions
    assert output_type.__bases__ == (ModelOutput,)
    assert next(iter(output_type.__dataclass_fields__)) == "last_hidden_state"


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


@pytest.mark.parametrize("kind", ["qwen2_attention", "qwen2_mlp", "qwen2_norm", "qwen2_layer", "qwen3_attention", "qwen3_mlp", "qwen3_norm", "qwen3_layer", "phi_attention", "phi_mlp", "phi_layer", "gpt2_attention", "gpt2_mlp", "gpt2_block"])
def test_decoder_family_contracts_reject_wrong_shape_or_dtype(kind: str) -> None:
    module = _module(kind)
    assert native_module_outputs(module, TensorSpec(["B", "S", 8], "F32", "hidden_states")) is None
    assert native_module_outputs(module, TensorSpec(["B", "S", 16], "F16", "hidden_states")) is None


@pytest.mark.parametrize("kind", ["qwen2_attention", "qwen2_mlp", "qwen2_layer", "qwen3_attention", "qwen3_mlp", "qwen3_layer", "phi_attention", "phi_mlp", "phi_layer", "gpt2_attention", "gpt2_mlp", "gpt2_block"])
def test_decoder_family_contracts_reject_modified_execution_metadata(kind: str) -> None:
    module = _module(kind)
    if kind.startswith(("qwen2", "qwen3", "phi")):
        attention = getattr(module, "self_attn", module)
        attention.config._attn_implementation = "sdpa"
    elif kind.startswith("gpt2"):
        if kind == "gpt2_mlp":
            module.act = nn.Identity()
        else:
            attention = getattr(module, "attn", module)
            attention.config._attn_implementation = "sdpa"
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize("kind", ["qwen2_layer", "qwen3_layer", "phi_layer", "gpt2_block"])
def test_decoder_family_contracts_reject_replaced_nested_children(kind: str) -> None:
    module = _module(kind)
    child_name = "input_layernorm" if kind.startswith(("qwen2", "qwen3", "phi")) else "ln_1"
    setattr(module, child_name, nn.Identity())
    assert native_module_outputs(module, SPEC) is None


@pytest.mark.parametrize(
    ("kind", "activation_name"),
    [
        ("qwen2_mlp", "act_fn"),
        ("qwen3_mlp", "act_fn"),
        ("phi_mlp", "activation_fn"),
        ("gpt2_mlp", "act"),
    ],
)
def test_decoder_family_contracts_reject_overridden_gelu_callable(kind: str, activation_name: str) -> None:
    module = _module(kind)
    activation = GELUActivation().eval()
    activation.act = lambda value: value
    setattr(module, activation_name, activation)
    assert native_module_outputs(module, SPEC) is None


def _native_forward_returns_tuple(cls: type) -> bool:
    function = ast.parse(textwrap.dedent(inspect.getsource(cls.forward))).body[0]
    returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    returned = returns[0]
    if isinstance(returned, ast.Tuple):
        return True
    if isinstance(returned, ast.Name):
        return any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == returned.id for target in node.targets)
            and isinstance(node.value, ast.Tuple)
            for node in ast.walk(function)
        )
    raise AssertionError(f"Unreviewed native return form: {ast.dump(returned)}")


@pytest.mark.parametrize(
    ("cls", "is_tuple"),
    [
        (qwen2.Qwen2Attention, True),
        (qwen2.Qwen2MLP, False),
        (qwen2.Qwen2DecoderLayer, False),
        (qwen3.Qwen3Attention, True),
        (qwen3.Qwen3MLP, False),
        (qwen3.Qwen3DecoderLayer, False),
        (phi.PhiAttention, True),
        (phi.PhiMLP, False),
        (phi.PhiDecoderLayer, True),
        (gpt2.GPT2Attention, True),
        (gpt2.GPT2MLP, False),
        (gpt2.GPT2Block, True),
    ],
)
def test_decoder_family_output_ports_follow_native_return_ast(cls: type, is_tuple: bool) -> None:
    assert _native_forward_returns_tuple(cls) is is_tuple
