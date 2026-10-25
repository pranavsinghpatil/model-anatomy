"""Facts must follow proven operators, casts and feasible control-flow paths."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from model_anatomy.src.static_flow import StaticFlowAnalyzer


def mystery(value):
    raise AssertionError("Static analysis must never execute a source function")


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(
            hidden_size=4, num_attention_heads=1, is_encoder_decoder=False, apply_spec_augment=False
        )
        self.proj = torch.nn.Linear(4, 8, device="meta")
        self.left = torch.nn.Identity()
        self.right = torch.nn.Identity()
        self.eval()


class Chunk(Base):
    def forward(self, hidden_states):
        first, second = self.proj(hidden_states).chunk(2, dim=-1)
        return first * second


class CastGuard(Base):
    def forward(self, hidden_states):
        value = hidden_states.half()
        if value.dtype == torch.float16:
            return self.left(value)
        return self.right(value)


class FloatGuard(Base):
    def forward(self, hidden_states):
        if hidden_states.dtype == torch.float16:
            return self.left(hidden_states)
        return self.right(hidden_states)


class UnknownGuard(Base):
    def forward(self, hidden_states):
        value = mystery(hidden_states)
        if value.dtype == torch.float16:
            return self.left(value)
        return self.right(value)


class JoinedGuard(Base):
    def forward(self, hidden_states):
        if mystery(hidden_states):
            value = hidden_states.half()
        else:
            value = hidden_states.float()
        if value.dtype == torch.float16:
            return self.left(value)
        return self.right(value)


class ShapeMetadata(Base):
    def forward(self, hidden_states):
        shape = hidden_states.size()
        return hidden_states.view((-1,) + shape[1:]).to(hidden_states.device)


class ConfigGuard(Base):
    def forward(self, hidden_states):
        if not getattr(self.config, "apply_spec_augment", True):
            return self.left(hidden_states)
        return self.right(hidden_states)


class ZipValidation(Base):
    def forward(self, hidden_states):
        for mask, name in zip([None, None], ["head", "cross"]):
            if mask is not None:
                hidden_states = mystery(hidden_states)
        return self.left(hidden_states)


class RangeLayers(Base):
    def forward(self, hidden_states):
        for index in range(2):
            hidden_states = self.left(hidden_states)
        return hidden_states


class TooLargeRange(Base):
    def forward(self, hidden_states):
        for index in range(1_000_000):
            hidden_states = self.left(hidden_states)
        return hidden_states


class ShadowedTorch(Base):
    def forward(self, hidden_states, torch=None):
        return torch.isinf(hidden_states)


class PositionRange(Base):
    def forward(self, hidden_states):
        return torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)


class StringView(Base):
    def forward(self, hidden_states):
        return hidden_states.view("B", "S", 4)


class UnknownNoneGuard(Base):
    def forward(self, hidden_states):
        value = mystery(hidden_states)
        if value is None:
            return self.left(hidden_states)
        return self.right(hidden_states)


class NativeTensorGuard(Base):
    def forward(self, hidden_states):
        if isinstance(self.proj.weight, torch.Tensor):
            return self.left(hidden_states)
        return self.right(hidden_states)


def analyze(model):
    return StaticFlowAnalyzer(
        model, "", {path: SimpleNamespace(kind="backbone") for path, _ in model.named_modules()}
    ).analyze()


def paths(flow):
    return [event.module_path for event in flow.events if event.module_path]


def test_chunk_outputs_have_individual_shapes_and_recombine_with_proven_facts():
    flow = analyze(Chunk())
    split = next(event for event in flow.events if event.kind == "split")
    assert flow.warnings == []
    assert split.metadata["output_arity"] == 2
    assert [spec["shape"] for spec in split.metadata["output_port_specs"].values()] == [["B", "S", 4]] * 2
    assert flow.outputs.spec.shape == ["B", "S", 4]
    assert flow.outputs.spec.dtype == "F32"


def test_cast_updates_dtype_before_guard_and_canonical_f32_prunes_only_proven_branch():
    cast = analyze(CastGuard())
    original = analyze(FloatGuard())
    assert paths(cast) == ["left"] and cast.outputs.spec.dtype == "F16"
    assert paths(original) == ["right"] and original.outputs.spec.dtype == "F32"
    assert cast.warnings == original.warnings == []


def test_unknown_transform_and_conflicting_branch_dtypes_cannot_become_known():
    for model in (UnknownGuard(), JoinedGuard()):
        flow = analyze(model)
        assert paths(flow) == ["left", "right"]
        assert flow.outputs.spec is None
        assert any("Conditional source path is unresolved" in warning for warning in flow.warnings)


def test_size_tuple_slice_concat_and_device_copy_preserve_only_proven_shape():
    flow = analyze(ShapeMetadata())
    assert flow.warnings == []
    assert flow.outputs.spec.shape == ["B", "S", 4]
    assert flow.outputs.spec.dtype == "F32"
    assert any(item.get("positive_symbolic_axes") == ["B", "S"] for item in flow.diagnostics)


def test_native_config_getattr_and_bounded_zip_resolve_metadata_without_execution():
    for model in (ConfigGuard(), ZipValidation()):
        flow = analyze(model)
        assert flow.warnings == [] and paths(flow) == ["left"]


def test_finite_range_unrolls_under_budget_and_oversized_range_remains_unresolved():
    bounded = analyze(RangeLayers())
    oversized = analyze(TooLargeRange())
    assert bounded.warnings == [] and paths(bounded) == ["left", "left"]
    assert any("Loop could not be statically resolved" in warning for warning in oversized.warnings)


def test_torch_name_shadowing_cannot_authorize_an_operator_contract():
    flow = analyze(ShadowedTorch())
    assert not any(event.metadata.get("contract") for event in flow.events)
    assert any("Opaque call" in warning for warning in flow.warnings)


def test_zero_based_integer_range_proves_its_own_rank_before_unsqueeze():
    flow = analyze(PositionRange())
    assert flow.warnings == []
    assert "position_ids" not in flow.inputs
    assert flow.outputs.spec.shape == [1, "S"]
    assert flow.outputs.spec.dtype == "I64"


def test_literal_strings_are_not_symbolic_reshape_dimensions():
    flow = analyze(StringView())
    assert flow.outputs.spec.shape is None
    assert any("view has no proven output contract" in warning for warning in flow.warnings)


def test_unknown_call_can_return_none_and_does_not_prune_either_branch():
    flow = analyze(UnknownNoneGuard())
    assert paths(flow) == ["left", "right"]
    assert any("Conditional source path is unresolved" in warning for warning in flow.warnings)


def test_tensor_instance_guard_uses_inspected_parameter_metadata():
    flow = analyze(NativeTensorGuard())
    assert paths(flow) == ["left"] and flow.warnings == []


@pytest.mark.parametrize(
    "operation",
    [torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.GELU(), torch.nn.LayerNorm(4, elementwise_affine=False)],
)
def test_integer_activation_inputs_do_not_inherit_an_integer_output_contract(operation):
    model = Base()
    model.left = operation
    analyzer = StaticFlowAnalyzer(model, "", {"": SimpleNamespace(kind="backbone")})
    from model_anatomy.src.ir import TensorSpec
    from model_anatomy.src.static_flow import StaticValue

    assert analyzer.module_spec("left", {"0": StaticValue(spec=TensorSpec([2, 4], "I64", "tokens"))}) is None
