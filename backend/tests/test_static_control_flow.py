"""Try statements retain supported facts while reporting omitted paths."""

from types import SimpleNamespace

import torch

from model_anatomy.src.static_flow import StaticFlowAnalyzer


class TryWithHandler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.layer = torch.nn.Linear(2, 2, device="meta")

    def forward(self, x):
        try:
            value = self.layer(x)
        except RuntimeError:
            value = x + x
        return value


class TryWithElse(TryWithHandler):
    def forward(self, x):
        try:
            value = self.layer(x)
        except RuntimeError:
            value = x
        else:
            value = x + x
        return value


class TryWithFinally(TryWithHandler):
    def forward(self, x):
        try:
            return self.layer(x)
        finally:
            x = x + x


class TryReturnedBody(TryWithHandler):
    def forward(self, x):
        try:
            return self.layer(x)
        except RuntimeError:
            return x


class WithBody(TryWithHandler):
    def forward(self, x):
        with torch.no_grad():
            return self.layer(x)


def analyze(model):
    return StaticFlowAnalyzer(
        model, "", {"": SimpleNamespace(kind="backbone"), "layer": SimpleNamespace(kind="linear")}
    ).analyze()


def test_try_handlers_warn_and_keep_try_body_dependencies():
    flow = analyze(TryWithHandler())
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.outputs.refs == [(flow.events[0].id, "output")]
    assert any("exception handlers" in warning for warning in flow.warnings)


def test_try_else_warns_and_keeps_try_body_dependencies():
    flow = analyze(TryWithElse())
    assert [event.module_path for event in flow.events] == ["layer"]
    assert any("else path" in warning for warning in flow.warnings)


def test_try_finally_warns_even_when_main_body_returns():
    flow = analyze(TryWithFinally())
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.outputs.refs == [(flow.events[0].id, "output")]
    assert any("finally block" in warning for warning in flow.warnings)


def test_try_main_body_return_is_retained_and_handlers_are_reported():
    flow = analyze(TryReturnedBody())
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.outputs.refs == [(flow.events[0].id, "output")]
    assert any("exception handlers" in warning for warning in flow.warnings)


def test_with_body_remains_supported_without_try_warning():
    flow = analyze(WithBody())
    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.outputs.refs == [(flow.events[0].id, "output")]
    assert flow.warnings == []
