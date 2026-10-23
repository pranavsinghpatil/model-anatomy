"""Bind helper source arguments without executing model computations."""

from types import SimpleNamespace

import torch
from transformers.pytorch_utils import apply_chunking_to_forward

from model_anatomy.src.static_flow import StaticFlowAnalyzer


class Defaults(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.layer = torch.nn.Linear(2, 2, device="meta")

    def forward(self, x):
        return self.choose(x)

    def choose(self, x, dtype=None):
        if dtype is None:
            return self.layer(x)
        return x


class ReorderedKeywords(Defaults):
    def forward(self, x):
        return self.choose(second=self.layer(x), first=x)

    def choose(self, first, second):
        return first


class KeywordOnly(Defaults):
    def forward(self, x):
        return self.choose(x, right=self.layer(x), use_right=True)

    def choose(self, left, ignored=None, *, right=None, use_right=False):
        if use_right:
            return right
        return left


class Variadic(Defaults):
    def forward(self, x):
        return self.choose(x, self.layer(x), use_right=True)

    def choose(self, x, *rest, use_right=False):
        if use_right:
            return rest[0]
        return x


class VariadicKeywords(Defaults):
    def forward(self, x):
        return self.choose(ignored=self.layer(x), value=x)

    def choose(self, **values):
        return values["value"]


class UnknownStarred(Defaults):
    def forward(self, x):
        return self.choose(*x)


class UnknownKeywords(Defaults):
    def forward(self, x):
        return self.choose(**{"x": x})  # noqa: PIE804 - exercise source argument expansion


class InvalidCall(Defaults):
    def forward(self, x):
        return self.choose(x, missing=self.layer(x))


class StaticMethod(Defaults):
    def forward(self, x):
        return self.choose(x)

    @staticmethod
    def choose(x, *rest):
        return x


class ClassMethod(Defaults):
    def forward(self, x):
        return self.choose(x)

    @classmethod
    def choose(cls, x, *rest):
        return x


class NativeChunking(Defaults):
    def forward(self, x):
        return apply_chunking_to_forward(self.choose, 0, 1, x)


class NamedChunking(Defaults):
    def forward(self, x):
        return apply_chunking_to_forward(forward_fn=self.choose, chunk_size=0, chunk_dim=1, input_tensors=x)


class ExpandedChunking(Defaults):
    def forward(self, x):
        return apply_chunking_to_forward(self.choose, 0, 1, *x)


class ActiveChunking(Defaults):
    def forward(self, x):
        return apply_chunking_to_forward(self.choose, 2, 1, x)


class SingletonGuard(Defaults):
    def forward(self, x):
        return self.choose(x, enabled=False)

    def choose(self, x, *, enabled=True):
        if enabled is False:
            return self.layer(x)
        return x


def analyze(model):
    return StaticFlowAnalyzer(
        model, "", {"": SimpleNamespace(kind="backbone"), "layer": SimpleNamespace(kind="linear")}
    ).analyze()


def test_literal_defaults_resolve_only_the_bound_helper_branch():
    flow = analyze(Defaults())
    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.outputs.refs == [(flow.events[0].id, "output")]
    assert "conditional" not in flow.events[0].metadata


def test_keyword_order_does_not_change_helper_dependencies():
    flow = analyze(ReorderedKeywords())
    assert flow.warnings == []
    assert flow.outputs.refs == [("input:x", "output")]


def test_omitted_positional_and_keyword_only_defaults_keep_their_names():
    flow = analyze(KeywordOnly())
    assert flow.warnings == []
    assert flow.outputs.refs == [(flow.events[0].id, "output")]


def test_variadic_values_keep_their_actual_tuple_members():
    flow = analyze(Variadic())
    assert flow.warnings == []
    assert flow.outputs.refs == [(flow.events[0].id, "output")]


def test_variadic_keyword_values_keep_only_the_selected_mapping_member():
    flow = analyze(VariadicKeywords())
    assert flow.warnings == []
    assert flow.outputs.refs == [("input:x", "output")]


def test_unknown_expansion_keeps_dependencies_and_reports_unresolved_binding():
    for model in (UnknownStarred(), UnknownKeywords()):
        flow = analyze(model)
        assert flow.outputs.refs == [("input:x", "output")]
        assert any("Helper arguments could not be statically bound" in warning for warning in flow.warnings)


def test_invalid_helper_binding_is_not_treated_as_a_valid_source_path():
    flow = analyze(InvalidCall())
    assert set(flow.outputs.refs) == {("input:x", "output"), (flow.events[0].id, "output")}
    assert any("Helper arguments could not be statically bound" in warning for warning in flow.warnings)


def test_static_and_class_methods_do_not_gain_a_false_tensor_argument():
    static = analyze(StaticMethod())
    assert static.warnings == []
    assert static.outputs.refs == [("input:x", "output")]

    class_method = analyze(ClassMethod())
    assert class_method.warnings == [
        "Helper arguments could not be statically bound: class method .choose"
    ]
    assert class_method.outputs.refs == [("input:x", "output")]


def test_native_positional_chunking_preserves_its_tensor_argument():
    flow = analyze(NativeChunking())
    assert flow.warnings == []
    assert flow.outputs.refs == [(flow.events[0].id, "output")]


def test_unresolved_chunking_call_forms_keep_dependencies_and_do_not_guess():
    for model in (NamedChunking(), ExpandedChunking(), ActiveChunking()):
        flow = analyze(model)
        assert flow.outputs.refs == [("input:x", "output")]
        assert any("Chunked helper arguments are unresolved" in warning for warning in flow.warnings)


def test_bound_boolean_defaults_can_use_python_singleton_identity_guards():
    flow = analyze(SingletonGuard())
    assert flow.warnings == []
    assert flow.outputs.refs == [(flow.events[0].id, "output")]
