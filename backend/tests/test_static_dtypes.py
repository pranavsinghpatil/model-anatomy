from types import SimpleNamespace

import pytest
import torch

from model_anatomy.src.static_dtypes import cast_dtype, factory_dtype
from model_anatomy.src.static_flow import UNKNOWN, StaticFlowAnalyzer


class DtypeModel(torch.nn.Module):
    def __init__(self, parameter_dtype=torch.float32):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False, torch_dtype=parameter_dtype)
        self.weight = torch.nn.Parameter(torch.empty(1, device="meta", dtype=parameter_dtype))
        self.layer = torch.nn.Linear(1, 1, device="meta", dtype=parameter_dtype)

    def forward(self, hidden_states, input_ids=None, attention_mask=None):
        return hidden_states


class CastModel(DtypeModel):
    def forward(self, hidden_states):
        if hidden_states.half().dtype == torch.float16:
            return self.layer(hidden_states)
        return hidden_states.float()


class ExplicitToModel(DtypeModel):
    def forward(self, hidden_states):
        cast = hidden_states.to(dtype=torch.float16)
        return cast


class PositionalToModel(DtypeModel):
    def forward(self, hidden_states):
        return hidden_states.to("cpu", torch.float16)


class NoneToModel(DtypeModel):
    def forward(self, hidden_states):
        return hidden_states.to(dtype=None)


class OpaqueToModel(DtypeModel):
    def forward(self, hidden_states):
        other = hidden_states.nonzero()
        return hidden_states.to(other)


class UnknownSecondToModel(DtypeModel):
    def forward(self, hidden_states):
        target = torch.float16 if hidden_states.requires_grad else torch.float32
        return hidden_states.to("cpu", target)


class ExpandedToModel(DtypeModel):
    def forward(self, hidden_states):
        return hidden_states.to(**{"dtype": torch.float16})  # noqa: PIE804 - inspect unsupported expansion


class TargetDependencyModel(DtypeModel):
    def forward(self, hidden_states):
        return hidden_states.to(self.layer(hidden_states))


class FactoryMaskModel(DtypeModel):
    def forward(self, hidden_states):
        mask = torch.ones((2, 2), dtype=torch.bool)
        if mask.dtype == torch.bool:
            return self.layer(hidden_states)
        return hidden_states


class IntegerFactoryMaskModel(DtypeModel):
    def forward(self, hidden_states):
        mask = torch.full((2, 2), 0, dtype=None)
        if mask.dtype == torch.int64:
            return self.layer(hidden_states)
        return hidden_states


class PerParameterDtypeModel(DtypeModel):
    def __init__(self):
        super().__init__(torch.float32)
        self.layer = torch.nn.Linear(1, 1, device="meta", dtype=torch.float16)

    def forward(self, hidden_states):
        if self.layer.weight.dtype == torch.float16:
            return self.layer(hidden_states)
        return hidden_states


class JoinModel(DtypeModel):
    def forward(self, hidden_states):
        if hidden_states.requires_grad:
            value = hidden_states.half()
        else:
            value = hidden_states.float()
        return value


class SameJoinModel(DtypeModel):
    def forward(self, hidden_states):
        if hidden_states.requires_grad:
            value = hidden_states.half()
        else:
            value = hidden_states.to(torch.float16)
        return value


class OpaqueDtypeModel(DtypeModel):
    def forward(self, hidden_states):
        value = hidden_states.nonzero()
        if value.dtype == torch.int64:
            return self.layer(hidden_states)
        return hidden_states


class ShadowedDtypeModel(DtypeModel):
    def forward(self, hidden_states, torch=None):
        if hidden_states.to(dtype=torch.float16).dtype == torch.float16:
            return self.layer(hidden_states)
        return hidden_states


def analyze(model):
    paths = {
        "": SimpleNamespace(kind="backbone"),
        "layer": SimpleNamespace(kind="linear"),
    }
    return StaticFlowAnalyzer(model, "", paths).analyze()


def test_native_meta_dtype_oracle_is_attached_to_explicit_activation_role():
    model = DtypeModel(torch.float16)
    flow = analyze(model)
    assert flow.inputs["hidden_states"].dtype == next(model.parameters()).dtype
    assert flow.inputs["input_ids"].dtype == torch.int64
    attention_mask = StaticFlowAnalyzer(model, "", {"": SimpleNamespace(kind="backbone")}).input(
        "attention_mask"
    )
    assert attention_mask.dtype is not torch.float16


def test_half_cast_changes_dtype_before_guard_and_float32_input_does_not_take_else():
    flow = analyze(CastModel(torch.float32))
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]
    assert any(item["predicate"] == "tensor.dtype" for item in flow.profile_assumptions)


def test_parameter_dtype_uses_the_referenced_parameter_not_first_model_parameter():
    flow = analyze(PerParameterDtypeModel())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]


def test_explicit_to_dtype_is_preserved_on_returned_value():
    flow = analyze(ExplicitToModel())
    assert flow.outputs.dtype == torch.float16
    assert analyze(PositionalToModel()).outputs.dtype == torch.float16
    assert analyze(NoneToModel()).outputs.dtype == torch.float32
    assert analyze(OpaqueToModel()).outputs.dtype is not torch.float32


@pytest.mark.parametrize("model_type", [OpaqueToModel, UnknownSecondToModel, ExpandedToModel])
def test_unresolved_to_overloads_do_not_fall_back_to_input_dtype(model_type):
    assert analyze(model_type()).outputs.dtype is UNKNOWN


def test_to_target_dependencies_are_retained_without_duplicate_module_calls():
    flow = analyze(TargetDependencyModel())
    assert [event.module_path for event in flow.events] == ["layer"]
    assert (flow.events[0].id, "output") in flow.outputs.refs
    assert flow.outputs.dtype is UNKNOWN


def test_explicit_boolean_factory_dtype_controls_mask_branch_without_factory_execution():
    flow = analyze(FactoryMaskModel())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]


def test_integer_full_mask_dtype_is_inferred_from_fill_value():
    flow = analyze(IntegerFactoryMaskModel())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]


def test_join_preserves_only_agreeing_dtype_facts():
    assert analyze(SameJoinModel()).outputs.dtype == torch.float16
    assert analyze(JoinModel()).outputs.dtype is not torch.float16


def test_opaque_and_shadowed_dtype_accesses_stay_unknown():
    opaque = analyze(OpaqueDtypeModel())
    assert any("value.dtype == torch.int64" in warning for warning in opaque.warnings)
    shadowed = analyze(ShadowedDtypeModel())
    assert any("hidden_states.to(dtype=torch.float16).dtype" in warning for warning in shadowed.warnings)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64, torch.bool])
@pytest.mark.parametrize("method", ["float", "half", "double", "bfloat16"])
def test_cast_rule_matches_native_meta_tensor_dtype(dtype, method):
    value = torch.empty((2, 3), device="meta", dtype=dtype)
    assert cast_dtype(method, dtype) == getattr(value, method)().dtype


@pytest.mark.parametrize("target", [torch.float16, torch.bfloat16, torch.int64, torch.bool])
def test_to_and_type_as_rules_match_meta_overloads(target):
    value = torch.empty((2, 3), device="meta", dtype=torch.float32)
    prototype = torch.empty((2, 3), device="meta", dtype=target)
    assert cast_dtype("to", value.dtype, [target]) == value.to(target).dtype
    assert cast_dtype("to", value.dtype, ["meta", target]) == value.to("meta", target).dtype
    assert cast_dtype("type_as", value.dtype, type_as_dtype=target) == value.type_as(prototype).dtype
    assert (
        cast_dtype("to", value.dtype, [object()], positional_is_tensor=True, type_as_dtype=target)
        == value.to(prototype).dtype
    )


@pytest.mark.parametrize("default", [torch.float32, torch.float64])
@pytest.mark.parametrize("fill", [True, 7, 1.5])
def test_full_factory_rule_matches_native_fill_inference(default, fill):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default)
        oracle = torch.full((2, 3), fill, device="meta")
        assert factory_dtype("full", full_value=fill, full_value_known=True) == oracle.dtype
        assert factory_dtype("ones") == torch.ones((2, 3), device="meta").dtype
    finally:
        torch.set_default_dtype(previous)


def test_unknown_full_fill_and_unknown_to_target_have_no_dtype_fact():
    assert factory_dtype("full") is None
    assert factory_dtype("full", explicit_dtype_present=True) is None
    assert cast_dtype("to", torch.float32, [object()]) is None
    assert cast_dtype("to", torch.float32, ["cpu", object()]) is None
    assert cast_dtype("to", torch.float32, dtype_keyword_present=True) is None
