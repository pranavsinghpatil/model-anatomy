"""Torch functional transfers need callable identity and explicit operator evidence."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from model_anatomy.src.static_flow import StaticFlowAnalyzer


def dropout(x):
    raise AssertionError("A similarly named user function must not run")


class Functional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)

    def forward(self, x):
        x = F.dropout(x, p=0.25, training=False)
        x = F.relu(x)
        x = F.gelu(x, approximate="tanh")
        x = F.silu(x)
        x = torch.tanh(x)
        x = torch.sigmoid(x)
        return F.softmax(x, dim=-1)


class Shadowed(Functional):
    def forward(self, x):
        return dropout(x)


class ShadowedNamespace(Functional):
    def forward(self, x, F=None):
        return F.dropout(x)


class CastSoftmax(Functional):
    def forward(self, x):
        return F.softmax(x, dim=-1, dtype=torch.float64)


class InvalidFunctionalCall(Functional):
    def forward(self, x):
        return F.relu(x, unknown=True)


def analyze(model):
    return StaticFlowAnalyzer(model, "", {"": SimpleNamespace(kind="backbone")}).analyze()


def test_actual_functional_calls_form_a_connected_operation_chain_with_provenance():
    flow = analyze(Functional())
    assert flow.warnings == []
    assert [event.kind for event in flow.events] == ["dropout", *["activation"] * 5, "softmax"]
    assert flow.events[0].arguments["input"].refs == [("input:x", "output")]
    for left, right in zip(flow.events, flow.events[1:]):
        assert right.arguments["input"].refs == [(left.id, "output")]
    assert flow.outputs.refs == [(flow.events[-1].id, "output")]
    assert flow.events[0].metadata["parameters"] == {"p": 0.25, "training": False, "inplace": False}
    assert flow.events[2].metadata["parameters"] == {"approximate": "tanh"}
    assert all(event.metadata["runtime_verified"] is False for event in flow.events)
    assert all(event.metadata["callable"].startswith("torch.") for event in flow.events)


@pytest.mark.parametrize("model_class", [Shadowed, ShadowedNamespace, CastSoftmax, InvalidFunctionalCall])
def test_unproven_or_unsupported_functional_calls_keep_dependencies_and_warnings(model_class):
    flow = analyze(model_class())
    assert flow.events == []
    assert flow.outputs.refs == [("input:x", "output")]
    assert any("Opaque call" in warning for warning in flow.warnings)
