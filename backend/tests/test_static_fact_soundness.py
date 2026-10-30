"""Adversarial source cases must not acquire unproved tensor facts."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from model_anatomy.src.static_flow import StaticFlowAnalyzer


def arange(*_args, **_kwargs):
    raise AssertionError("Publisher helpers must not execute")


def cat(*_args, **_kwargs):
    raise AssertionError("Publisher helpers must not execute")


def unresolved_guard():
    raise AssertionError("Unknown guards must not execute")


class Envelope:
    def __init__(self, **_kwargs):
        raise AssertionError("Publisher constructors must not execute")


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="toy", is_encoder_decoder=False, hidden_size=4)


class FakeArange(Base):
    def forward(self):
        return arange(4)


class FloatArange(Base):
    def forward(self):
        return torch.arange(4, dtype=torch.float32)


class FakeCat(Base):
    def forward(self, x):
        return cat(x)


class MismatchedLayerNorm(Base):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(1, device="meta", dtype=torch.float32))
        self.norm = nn.LayerNorm(4, device="meta", dtype=torch.float16)

    def forward(self, x):
        return self.norm(x)


class HookedLinear(Base):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, device="meta")
        self.projection.register_forward_hook(lambda *_args: torch.empty(1, device="meta"))

    def forward(self, x):
        return self.projection(x)


class ConditionalField(Base):
    def forward(self, x):
        if unresolved_guard():
            return Envelope(last_hidden_state=x)
        return x


class ShortCircuitedCall(Base):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, device="meta")

    def forward(self, x):
        if False and self.projection(x):  # noqa: SIM223 - exercise source short-circuit analysis
            return x
        return x


class ConditionalExpressionCall(Base):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, device="meta")

    def forward(self, x):
        return self.projection(x) if unresolved_guard() else x


class ConditionalBothArms(Base):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 4, device="meta")
        self.right = nn.Linear(4, 4, device="meta")

    def forward(self, x):
        return self.left(x) if unresolved_guard() else self.right(x)


class ConditionalStatementArms(ConditionalBothArms):
    def forward(self, x):
        if unresolved_guard():
            result = self.left(x)
        else:
            result = self.right(x)
        return result


class PublisherProperty(Base):
    @property
    def trigger(self):
        raise AssertionError("Publisher property must not execute")

    def forward(self, x):
        if self.trigger:
            return x
        return x


class StarredView(Base):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(1, device="meta"))

    def forward(self, x):
        batch = (x.size(0),)
        rest = (x.size(1), 4)
        return x.view(*batch, *rest)


def unknown_shape(*_args):
    return None


class UnknownStarredView(StarredView):
    def forward(self, x):
        return x.view(*unknown_shape(x))


def analyze(module: nn.Module):
    kinds = {name: SimpleNamespace(kind="backbone") for name, _ in module.named_modules()}
    return StaticFlowAnalyzer(module.eval(), "", kinds).analyze()


def test_same_named_arange_helper_does_not_invent_position_ids() -> None:
    flow = analyze(FakeArange())
    assert "position_ids" not in flow.inputs
    assert flow.outputs.spec is None


def test_torch_arange_with_explicit_float_dtype_is_not_integer_position_ids() -> None:
    flow = analyze(FloatArange())
    assert "position_ids" not in flow.inputs
    assert flow.outputs.spec is None or flow.outputs.spec.dtype != "I64"


def test_same_named_cat_helper_is_not_reported_as_torch_concat() -> None:
    flow = analyze(FakeCat())
    assert all(event.kind != "concatenation" for event in flow.events)


def test_mismatched_layer_norm_parameters_cannot_prove_output_dtype() -> None:
    flow = analyze(MismatchedLayerNorm())
    assert len(flow.events) == 1
    assert "output_port_specs" not in flow.events[0].metadata
    assert flow.outputs.spec is None


def test_forward_hook_blocks_native_linear_contract() -> None:
    flow = analyze(HookedLinear())
    assert len(flow.events) == 1
    assert "output_port_specs" not in flow.events[0].metadata
    assert flow.outputs.spec is None


def test_branch_join_keeps_only_fields_present_on_every_return() -> None:
    flow = analyze(ConditionalField())
    assert not flow.outputs.fields


def test_constant_short_circuit_does_not_add_unreachable_module_call() -> None:
    flow = analyze(ShortCircuitedCall())
    assert flow.events == []


def test_unknown_conditional_expression_marks_call_as_conditional() -> None:
    flow = analyze(ConditionalExpressionCall())
    assert len(flow.events) == 1
    assert flow.events[0].metadata.get("conditional") == "unresolved_guard()"


def test_conditional_expression_arms_have_opposite_conditions() -> None:
    flow = analyze(ConditionalBothArms())
    assert [event.metadata.get("conditional") for event in flow.events] == [
        "unresolved_guard()",
        "not (unresolved_guard())",
    ]


def test_conditional_statement_arms_have_opposite_conditions() -> None:
    flow = analyze(ConditionalStatementArms())
    assert [event.metadata.get("conditional") for event in flow.events] == [
        "unresolved_guard()",
        "not (unresolved_guard())",
    ]


def test_publisher_property_is_not_invoked_during_static_analysis() -> None:
    flow = analyze(PublisherProperty())
    assert flow.outputs.refs == [("input:x", "output")]


def test_starred_view_uses_only_static_shape_items() -> None:
    flow = analyze(StarredView())
    assert flow.warnings == []
    assert flow.outputs.spec.shape == ["B", "S", 4]


def test_unknown_starred_view_does_not_claim_an_output_shape() -> None:
    flow = analyze(UnknownStarredView())
    assert flow.outputs.spec is not None
    assert flow.outputs.spec.shape is None
    assert flow.outputs.spec.dtype == "F32"
    assert flow.warnings[-1] == "Tensor transform view has no proven output contract"
