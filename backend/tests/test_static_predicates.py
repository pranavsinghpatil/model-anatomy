"""Canonical tracing guards need identity evidence; similarly named calls stay opaque."""

from types import SimpleNamespace

import pytest
import torch
from transformers.utils.import_utils import is_torch_fx_proxy as native_fx_predicate
from transformers.utils.import_utils import is_torchdynamo_compiling

from model_anatomy.src.static_flow import StaticFlowAnalyzer


def is_torch_fx_proxy(x):
    raise AssertionError("Model helpers must never execute")


class NativeGuards(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.layer = torch.nn.Linear(2, 2, device="meta")

    def forward(self, x):
        return self.padding_check(x)

    def padding_check(self, x):
        tracing = native_fx_predicate(x) or torch.jit.is_tracing() or is_torchdynamo_compiling()
        if tracing:
            result = x + x
        else:
            result = self.layer(x)
        return result


class OpaqueGuard(NativeGuards):
    def forward(self, x):
        if is_torch_fx_proxy(x):
            result = self.layer(x)
        else:
            result = x + x
        return result


class ShadowedGuard(NativeGuards):
    def forward(self, x, native_fx_predicate=None):
        if native_fx_predicate(x):
            result = self.layer(x)
        else:
            result = x + x
        return result


def unresolved_guard():
    raise AssertionError("Unknown predicates must not execute")


class ConditionalReturns(NativeGuards):
    def forward(self, x):
        if unresolved_guard():
            return x[0]
        return x[1]


class TwoConditionalReturns(NativeGuards):
    def forward(self, x):
        if unresolved_guard():
            return x[0]
        else:
            return x[1]
        return self.layer(x)


class KnownEarlyReturn(NativeGuards):
    def forward(self, x):
        if not native_fx_predicate(x):
            return self.layer(x)
        return x + x


class ConditionalAssignment(NativeGuards):
    def forward(self, x):
        if unresolved_guard():
            x = x[0]
        return x


class ConditionalContinuation(NativeGuards):
    def forward(self, x):
        if unresolved_guard():
            return x
        return self.layer(x)


class ElseConditionalContinuation(NativeGuards):
    def forward(self, x):
        if unresolved_guard():
            x = x[0]
        else:
            return x
        return self.layer(x)


class NamedMaskGuard(NativeGuards):
    def forward(self, x, attention_mask=None):
        if not isinstance(mask := attention_mask, dict):
            mask = x
        return self.layer(mask)


class ShadowedBuiltinGuard(NativeGuards):
    def forward(self, x, isinstance=None):
        if not isinstance(mask := x, dict):
            mask = x[0]
        return self.layer(mask)


class DecoratedGuard(NativeGuards):
    @torch.no_grad()
    def forward(self, x):
        if native_fx_predicate(x):
            return x + x
        return self.layer(x)


def analyze(model):
    return StaticFlowAnalyzer(
        model, "", {"": SimpleNamespace(kind="backbone"), "layer": SimpleNamespace(kind="linear")}
    ).analyze()


def test_native_guards_are_source_audited_and_resolve_only_canonical_branch():
    flow = analyze(NativeGuards())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]
    assert all("conditional" not in event.metadata for event in flow.events)
    assert len(flow.profile_assumptions) == 3
    assert all(
        item["value"] is False and len(item["source_sha256"]) == 64 for item in flow.profile_assumptions
    )
    assert any("is_torch_fx_proxy" in item["predicate"] for item in flow.profile_assumptions)


@pytest.mark.parametrize("model_class", [OpaqueGuard, ShadowedGuard])
def test_same_name_or_shadowed_predicates_remain_unknown_without_execution(model_class):
    flow = analyze(model_class())
    assert flow.profile_assumptions == []
    assert any("Opaque call" in warning for warning in flow.warnings)
    assert any("Conditional source path is unresolved" in warning for warning in flow.warnings)
    assert len(flow.events) == 2
    assert all("conditional" in event.metadata for event in flow.events)


def test_tiny_native_gpt2_no_longer_has_an_opaque_padding_guard():
    from model_anatomy.src.registry import instantiate_native, load_native_config

    model, _ = instantiate_native(
        load_native_config(
            {
                "model_type": "gpt2",
                "architectures": ["GPT2LMHeadModel"],
                "vocab_size": 32,
                "n_positions": 16,
                "n_embd": 8,
                "n_head": 2,
                "n_layer": 2,
            }
        )
    )
    paths = {path: SimpleNamespace(kind="backbone") for path, _ in model.named_modules()}
    flow = StaticFlowAnalyzer(model, "transformer", paths).analyze()
    # Every native block now proves the same primary port, so the downstream
    # view is resolved without borrowing a default hidden-state shape.
    assert not any("padding" in warning or "is_torch_fx_proxy" in warning for warning in flow.warnings)
    assert not any("view has no proven output contract" in warning for warning in flow.warnings)
    assert any(
        event.metadata.get("contract") == "reviewed_native_module_list_member.v1"
        and event.metadata["output_port_specs"]["0"]["shape"] == ["B", "S", 8]
        for event in flow.events
    )
    assert any("is_torch_fx_proxy" in item["predicate"] for item in flow.profile_assumptions)


@pytest.mark.parametrize("model_class", [ConditionalReturns, TwoConditionalReturns])
def test_unknown_zero_argument_guard_retains_both_returns_and_reports_uncertainty(model_class):
    flow = analyze(model_class())
    assert flow.outputs.refs == [("input:x", "0"), ("input:x", "1")]
    assert any("Conditional source path is unresolved" in warning for warning in flow.warnings)
    assert flow.events == []


def test_known_early_return_does_not_analyze_unreachable_statements():
    flow = analyze(KnownEarlyReturn())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]


def test_unknown_assignment_only_branch_reports_uncertainty_and_keeps_dependencies():
    flow = analyze(ConditionalAssignment())
    assert set(flow.outputs.refs) == {("input:x", "0"), ("input:x", "output")}
    assert any("Conditional source path is unresolved" in warning for warning in flow.warnings)


@pytest.mark.parametrize(
    ("model_class", "condition"),
    [
        (ConditionalContinuation, "not (unresolved_guard())"),
        (ElseConditionalContinuation, "unresolved_guard()"),
    ],
)
def test_calls_after_one_sided_return_remain_conditional(model_class, condition):
    flow = analyze(model_class())
    assert any("Conditional source path is unresolved" in warning for warning in flow.warnings)
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.events[0].metadata["conditional"] == condition
    assert flow.events[0].arguments["0"].refs == [
        ("input:x", "0" if model_class is ElseConditionalContinuation else "output")
    ]


def test_named_assignment_in_native_mask_guard_is_resolved_from_canonical_none():
    flow = analyze(NamedMaskGuard())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.events[0].arguments["0"].refs == [("input:x", "output")]


def test_shadowed_builtin_guard_stays_uncertain():
    flow = analyze(ShadowedBuiltinGuard())
    assert any("Conditional source path is unresolved" in warning for warning in flow.warnings)
    assert set(flow.events[0].arguments["0"].refs) == {("input:x", "0"), ("input:x", "output")}


def test_decorated_forward_uses_original_source_globals_without_execution():
    flow = analyze(DecoratedGuard())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]
    assert len(flow.profile_assumptions) == 1
