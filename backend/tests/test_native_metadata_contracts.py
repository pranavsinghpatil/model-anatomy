from __future__ import annotations

import ast
from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig, T5Config
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.t5.modeling_t5 import T5Stack

from model_anatomy.src import native_metadata_contracts
from model_anatomy.src.ir import TensorSpec
from model_anatomy.src.native_metadata_contracts import (
    native_config_attribute,
    native_module_attribute,
    trusted_t5_module_utils_helper,
    trusted_t5_stack_method,
    trusted_torch_tensor_method,
)
from model_anatomy.src.static_flow import StaticFlowAnalyzer, StaticValue


def test_attention_implementation_reads_only_reviewed_backing_field() -> None:
    config = T5Config()
    vars(config)["_attn_implementation_internal"] = "sdpa"

    assert native_config_attribute(config, "_attn_implementation") == (True, "sdpa")


def test_missing_or_unreviewed_config_metadata_stays_unknown() -> None:
    config = T5Config()
    assert native_config_attribute(config, "_attn_implementation") == (False, None)
    assert native_config_attribute(config, "model_type") == (False, None)


def test_source_fingerprint_mismatch_disables_contract(monkeypatch) -> None:
    config = T5Config()
    vars(config)["_attn_implementation_internal"] = "sdpa"
    monkeypatch.setitem(
        native_metadata_contracts.CONTRACT_SOURCE_SHA256,
        "transformers.configuration_utils",
        "0" * 64,
    )

    assert native_config_attribute(config, "_attn_implementation") == (False, None)


def test_overridden_property_is_not_invoked() -> None:
    class UnsafeConfig(PretrainedConfig):
        @property
        def _attn_implementation(self):
            raise AssertionError("descriptor must not run")

    config = object.__new__(UnsafeConfig)
    assert native_config_attribute(config, "_attn_implementation") == (False, None)


def _meta_t5_stack() -> T5Stack:
    config = T5Config(d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2)
    with torch.device("meta"):
        return T5Stack(config, nn.Embedding(config.vocab_size, config.d_model)).eval()


def test_t5_module_dtype_is_derived_from_homogeneous_meta_parameters() -> None:
    module = _meta_t5_stack()
    assert native_module_attribute(module, "dtype") == (True, torch.float32)
    assert native_module_attribute(module, "training") == (False, None)


def test_t5_module_dtype_rejects_mixed_or_real_parameter_metadata() -> None:
    module = _meta_t5_stack()
    old = module.block[0].layer[0].SelfAttention.q.weight
    module.block[0].layer[0].SelfAttention.q.weight = nn.Parameter(
        torch.empty(old.shape, device="meta", dtype=torch.float16)
    )
    assert native_module_attribute(module, "dtype") == (False, None)

    module = _meta_t5_stack()
    old = module.block[0].layer[0].SelfAttention.q.weight
    module.block[0].layer[0].SelfAttention.q.weight = nn.Parameter(torch.empty(old.shape, dtype=torch.float32))
    assert native_module_attribute(module, "dtype") == (False, None)


def test_t5_module_dtype_rejects_overridden_traversal_or_xla_override(monkeypatch) -> None:
    module = _meta_t5_stack()
    module.parameters = lambda: iter(())
    assert native_module_attribute(module, "dtype") == (False, None)

    module = _meta_t5_stack()
    monkeypatch.setenv("XLA_USE_BF16", "true")
    assert native_module_attribute(module, "dtype") == (False, None)


def test_t5_module_dtype_rejects_import_time_xla_flag_and_base_traversal_patch(monkeypatch) -> None:
    module = _meta_t5_stack()
    monkeypatch.setitem(native_metadata_contracts._MODULE_UTILS_DTYPE_GETTER.__globals__, "XLA_DOWNCAST_BF16", "TRUE")
    assert native_module_attribute(module, "dtype") == (False, None)
    monkeypatch.undo()
    module = _meta_t5_stack()
    monkeypatch.setattr(nn.Module, "parameters", lambda self: iter(()))
    assert native_module_attribute(module, "dtype") == (False, None)


def test_t5_module_dtype_rejects_replaced_base_descriptor(monkeypatch) -> None:
    module = _meta_t5_stack()
    monkeypatch.setattr(ModuleUtilsMixin, "dtype", ModuleUtilsMixin.device)
    assert native_module_attribute(module, "dtype") == (False, None)


def test_t5_module_dtype_rejects_source_drift(monkeypatch) -> None:
    module = _meta_t5_stack()
    monkeypatch.setitem(
        native_metadata_contracts.CONTRACT_SOURCE_SHA256,
        "transformers.modeling_utils",
        "0" * 64,
    )
    assert native_module_attribute(module, "dtype") == (False, None)


def test_t5_tensor_contract_requires_exact_native_stack_methods(monkeypatch) -> None:
    module = _meta_t5_stack()
    assert trusted_t5_stack_method(module, "forward", T5Stack.forward)
    assert trusted_t5_stack_method(
        module,
        "_prepare_4d_causal_attention_mask_with_cache_position",
        T5Stack._prepare_4d_causal_attention_mask_with_cache_position,
    )

    def replacement(self, *args, **kwargs):
        return None

    monkeypatch.setattr(T5Stack, "forward", replacement)
    assert not trusted_t5_stack_method(module, "forward", replacement)


def test_t5_module_utils_helper_is_identity_and_source_guarded(monkeypatch) -> None:
    module = _meta_t5_stack()
    method = ModuleUtilsMixin.invert_attention_mask
    assert trusted_t5_module_utils_helper(module, "invert_attention_mask", method)
    monkeypatch.setattr(ModuleUtilsMixin, "invert_attention_mask", lambda self, mask: mask)
    assert not trusted_t5_module_utils_helper(
        module, "invert_attention_mask", ModuleUtilsMixin.invert_attention_mask
    )


def test_t5_tensor_contract_requires_native_torch_descriptors(monkeypatch) -> None:
    assert trusted_torch_tensor_method("clone")
    monkeypatch.setattr(torch.Tensor, "clone", lambda self: self)
    assert not trusted_torch_tensor_method("clone")


def test_t5_native_flow_resolves_dtype_finfo_and_masked_fill() -> None:
    module = _meta_t5_stack()
    classifications = {"": SimpleNamespace(kind="backbone")}
    analyzer = StaticFlowAnalyzer(module, "", classifications)
    analyzer.source_method_name = "forward"
    analyzer.source_method = T5Stack.forward
    analyzer.source_globals = T5Stack.forward.__globals__
    analyzer.env.update(
        {
            "hidden_states": StaticValue(
                [("input:hidden_states", "output")], spec=TensorSpec(["B", 1, 1, "S"], "F32", "hidden")
            ),
            "mask": StaticValue(
                [("input:mask", "output")], spec=TensorSpec(["B", 1, 1, "S"], "BOOL", "mask")
            ),
        }
    )

    minimum = analyzer.expression(ast.parse("torch.finfo(self.dtype).min", mode="eval").body)
    assert minimum.constant == -3.4028234663852886e38

    result = analyzer.expression(
        ast.parse("hidden_states.masked_fill(mask, torch.finfo(self.dtype).min)", mode="eval").body
    )
    assert result.spec is not None
    assert result.spec.shape == ["B", 1, 1, "S"]
    assert result.spec.dtype == "F32"


def test_t5_invert_attention_mask_tracks_native_dtype_conversion() -> None:
    module = _meta_t5_stack()
    analyzer = StaticFlowAnalyzer(module, "", {"": SimpleNamespace(kind="backbone")})
    analyzer.env["mask"] = StaticValue(
        [("input:mask", "output")], spec=TensorSpec(["B", "S"], "I64", "mask")
    )

    result = analyzer.expression(
        ast.parse("self.invert_attention_mask(mask)", mode="eval").body
    )
    cast = next(event for event in analyzer.flow.events if event.metadata.get("method") == "cast")
    assert cast.metadata["output_port_specs"]["output"] == {
        "shape": ["B", 1, 1, "S"],
        "dtype": "F32",
        "semantic_name": "cast",
    }
    assert result.refs


def _t5_mask_analyzer() -> StaticFlowAnalyzer:
    module = _meta_t5_stack()
    analyzer = StaticFlowAnalyzer(module, "", {"": SimpleNamespace(kind="decoder")})
    analyzer.source_method_name = "_prepare_4d_causal_attention_mask_with_cache_position"
    analyzer.source_method = T5Stack._prepare_4d_causal_attention_mask_with_cache_position
    analyzer.source_globals = analyzer.source_method.__globals__
    analyzer.positive_symbols.update({"B", "T"})
    analyzer.env.update({
        "sequence_length": StaticValue(symbol="T"),
        "target_length": StaticValue(symbol="T"),
        "batch_size": StaticValue(symbol="B"),
        "dtype": StaticValue(constant=torch.float32),
        "cache_position": StaticValue([("input:cache_position", "output")], spec=TensorSpec(["T"], "I64", "position")),
        "attention_mask": StaticValue([("input:attention_mask", "output")], spec=TensorSpec(["B", "T"], "I64", "mask")),
    })
    return analyzer


def test_t5_mask_producer_chain_and_full_slice_update_reach_return() -> None:
    analyzer = _t5_mask_analyzer()
    source = """
min_dtype = torch.finfo(dtype).min
causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device)
causal_mask = torch.triu(causal_mask, diagonal=1)
causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
causal_mask = causal_mask.clone()
mask_length = attention_mask.shape[-1]
padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
padding_mask = padding_mask == 0
causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
return causal_mask
"""
    analyzer.statements(ast.parse(source).body)
    events = {event.metadata.get("method"): event for event in analyzer.flow.events}
    for method in ("full", "triu", "gt", "float_bool_mul", "expand", "clone", "eq_scalar", "masked_fill"):
        assert method in events, (list(events), analyzer.flow.warnings)
        assert events[method].metadata["output_port_specs"]["output"]["shape"] is not None
    assert analyzer.flow.outputs.spec is not None
    assert analyzer.flow.outputs.spec.shape == ["B", 1, "T", "T"]
    assert analyzer.flow.outputs.refs == analyzer.env["causal_mask"].refs
    assert analyzer.flow.outputs.refs == [(events["masked_fill"].id, "output")]
    assert not analyzer.flow.warnings


def test_t5_mask_contract_rejects_bad_full_and_unreviewed_source(monkeypatch) -> None:
    analyzer = _t5_mask_analyzer()
    for expression in (
        "torch.full((sequence_length, target_length), fill_value=float('inf'), dtype=dtype, device=cache_position.device)",
        f"torch.full((sequence_length, target_length), fill_value={10**1000}, dtype=dtype, device=cache_position.device)",
        "torch.full((sequence_length, target_length), fill_value=0, dtype=torch.int64, device=cache_position.device)",
        "torch.full((sequence_length, target_length), fill_value=0, dtype=dtype, device='cpu')",
    ):
        assert analyzer.expression(ast.parse(expression, mode="eval").body).spec is None
    valid_full = "torch.full((sequence_length, target_length), fill_value=0, dtype=dtype, device=cache_position.device)"
    assert analyzer.expression(ast.parse(valid_full, mode="eval").body).spec is not None
    analyzer.env["float_mask"] = StaticValue([("input:float_mask", "output")], spec=TensorSpec(["B", "T"], "F32", "mask"))
    assert analyzer.expression(ast.parse("1.0 - float_mask", mode="eval").body).spec is not None
    from model_anatomy.src import native_output_contracts

    monkeypatch.setitem(native_output_contracts.CONTRACT_SOURCE_SHA256, "transformers.models.t5.modeling_t5", "0" * 64)
    assert analyzer.expression(ast.parse(valid_full, mode="eval").body).spec is None
    assert analyzer.expression(ast.parse("1.0 - float_mask", mode="eval").body).spec is None


def test_t5_full_contract_rejects_replaced_torch_function(monkeypatch) -> None:
    analyzer = _t5_mask_analyzer()
    monkeypatch.setattr(torch, "full", lambda *args, **kwargs: None)
    result = analyzer.expression(ast.parse("torch.full((sequence_length, target_length), fill_value=0, dtype=dtype, device=cache_position.device)", mode="eval").body)
    assert result.spec is None


def test_t5_partial_slice_assignment_invalidates_base_facts() -> None:
    analyzer = _t5_mask_analyzer()
    analyzer.env["causal_mask"] = StaticValue([("input:causal_mask", "output")], spec=TensorSpec(["B", 1, "T", "T"], "F32", "mask"))
    analyzer.env["replacement"] = StaticValue([("input:replacement", "output")], spec=TensorSpec(["B", 1, "T", "T"], "F32", "mask"))
    for source in ("causal_mask[:, :, :, :1] = replacement", "causal_mask[:, :, :, :True] = replacement", "causal_mask[:, :, :, :1.0] = replacement"):
        analyzer.env["causal_mask"] = StaticValue([("input:causal_mask", "output")], spec=TensorSpec(["B", 1, "T", "T"], "F32", "mask"))
        target = ast.parse(source).body[0].targets[0]
        analyzer.assign(target, analyzer.env["replacement"])
        assert analyzer.env["causal_mask"].spec is None
        assert "invalidated" in analyzer.flow.warnings[-1]


def test_t5_unknown_slice_bounds_do_not_become_full_slices() -> None:
    analyzer = _t5_mask_analyzer()
    analyzer.env["causal_mask"] = StaticValue([("input:causal_mask", "output")], spec=TensorSpec(["B", 1, "T", "T"], "F32", "mask"))
    for source in ("causal_mask[:, :, :, :unresolved]", "causal_mask[unresolved:, :, :, :]"):
        result = analyzer.expression(ast.parse(source, mode="eval").body)
        assert result.spec is None
    target = ast.parse("causal_mask[:, :, :, :unresolved] = causal_mask").body[0].targets[0]
    analyzer.assign(target, analyzer.env["causal_mask"])
    assert analyzer.env["causal_mask"].spec is None
    assert "invalidated" in analyzer.flow.warnings[-1]


class ConfigBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = T5Config()
        vars(self.config)["_attn_implementation_internal"] = "sdpa"

    def forward(self, x):
        if self.config._attn_implementation == "sdpa":
            return x
        return x


def test_static_flow_resolves_reviewed_config_property_without_warning() -> None:
    module = ConfigBranch()
    classifications = {name: SimpleNamespace(kind="backbone") for name, _ in module.named_modules()}
    flow = StaticFlowAnalyzer(module.eval(), "", classifications).analyze()

    assert not any("Conditional source path is unresolved" in warning for warning in flow.warnings)
    assert any(item["code"] == "native_config_attribute" for item in flow.diagnostics)
