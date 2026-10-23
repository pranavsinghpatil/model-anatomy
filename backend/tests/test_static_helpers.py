from types import SimpleNamespace

import pytest
import torch

from model_anatomy.src.static_flow import StaticFlowAnalyzer


class StaticHelperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.layer = torch.nn.Linear(2, 2, device="meta")

    @staticmethod
    def select(value, fallback=None, *, enabled=False):
        if fallback is None and not enabled:
            return value
        return value[0]

    def forward(self, x):
        return self.layer(self.select(x, enabled=False, fallback=None))


class InstanceHelperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.layer = torch.nn.Linear(2, 2, device="meta")

    def select(self, value, fallback=None, *, enabled=False):
        if fallback is None and not enabled:
            return value
        return value[0]

    def forward(self, x):
        return self.layer(self.select(x, enabled=False))


class StarredHelperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)

    def select(self, value):
        return value

    def forward(self, x):
        values = (x,)
        return self.select(*values)


class ExpandedKeywordHelperModel(StarredHelperModel):
    def forward(self, x):
        values = {"value": x}
        return self.select(**values)


class InvalidHelperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)

    def select(self, value, /):
        return value

    def forward(self, x):
        return self.select(value=x)


class MissingHelperArgumentModel(InvalidHelperModel):
    def forward(self, x):
        return self.select()


class ClassMethodHelperModel(InvalidHelperModel):
    @classmethod
    def select(cls, value):
        return value

    def forward(self, x):
        return self.select(x)


class ExpandedCallHelperModel(InstanceHelperModel):
    def forward(self, x):
        return self.select(**{"value": self.layer(x)})  # noqa: PIE804 - exercise AST expansion


class StarredCallHelperModel(InstanceHelperModel):
    def forward(self, x):
        return self.select(*(self.layer(x),))


class VariadicHelperModel(InvalidHelperModel):
    @staticmethod
    def select(*values):
        return values

    def forward(self, x):
        return self.select(x)


class KeywordRestHelperModel(InvalidHelperModel):
    @staticmethod
    def select(**values):
        return values

    def forward(self, x):
        return self.select(value=x)


def analyze(model):
    paths = {path: SimpleNamespace(kind="backbone") for path, _ in model.named_modules()}
    return StaticFlowAnalyzer(model, "", paths).analyze()


def test_static_helper_binds_first_positional_and_keyword_only_arguments():
    flow = analyze(StaticHelperModel())

    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.events[0].arguments["0"].refs == [("input:x", "output")]


def test_instance_helper_binds_omitted_defaults_and_keyword_only_arguments():
    flow = analyze(InstanceHelperModel())

    assert flow.warnings == []
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.events[0].arguments["0"].refs == [("input:x", "output")]


def test_starred_helper_arguments_warn_and_keep_their_dependencies():
    flow = analyze(StarredHelperModel())

    assert flow.warnings == ["Helper arguments could not be statically bound: self.select"]
    assert flow.outputs.refs == [("input:x", "output")]


def test_expanded_keyword_helper_arguments_warn_and_keep_their_dependencies():
    flow = analyze(ExpandedKeywordHelperModel())

    assert flow.warnings == ["Helper arguments could not be statically bound: self.select"]
    assert flow.outputs.refs == [("input:x", "output")]


def test_positional_only_parameter_cannot_be_bound_by_keyword():
    flow = analyze(InvalidHelperModel())

    assert flow.warnings == ["Helper arguments could not be statically bound: self.select"]
    assert flow.outputs.refs == [("input:x", "output")]


def test_missing_required_helper_argument_warns_without_inventing_a_binding():
    flow = analyze(MissingHelperArgumentModel())

    assert flow.warnings == ["Helper arguments could not be statically bound: self.select"]
    assert flow.outputs.refs == []


def test_classmethod_helper_is_reported_as_unresolved():
    flow = analyze(ClassMethodHelperModel())

    assert flow.warnings == ["Helper arguments could not be statically bound: class method .select"]
    assert flow.outputs.refs == [("input:x", "output")]


@pytest.mark.parametrize("model_type", [ExpandedCallHelperModel, StarredCallHelperModel])
def test_expansion_fallback_does_not_duplicate_argument_module_calls(model_type):
    flow = analyze(model_type())
    assert flow.warnings == ["Helper arguments could not be statically bound: self.select"]
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.outputs.refs == [(flow.events[0].id, "output")]


@pytest.mark.parametrize("model_type", [VariadicHelperModel, KeywordRestHelperModel])
def test_helper_rest_parameters_retain_input_dependencies(model_type):
    flow = analyze(model_type())
    assert flow.warnings == []
    assert flow.outputs.refs == [("input:x", "output")]
