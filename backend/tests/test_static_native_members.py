from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.gpt2 import modeling_gpt2 as gpt2
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from model_anatomy.src.static_flow import StaticFlowAnalyzer


def _config(width: int = 16) -> GPT2Config:
    config = GPT2Config(n_embd=width, n_head=2, n_layer=1)
    config._attn_implementation = "eager"
    return config


def _block(width: int = 16) -> gpt2.GPT2Block:
    return gpt2.GPT2Block(_config(width), layer_idx=0)


class LoopBlocks(nn.Module):
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.config = _config()
        with torch.device("meta"):
            self.layers = nn.ModuleList(members)

    def forward(self, hidden_states):
        for block in self.layers:
            hidden_states = block(hidden_states, use_cache=False, output_attentions=False)[0]
        return hidden_states


class DirectContainerCall(LoopBlocks):
    def forward(self, hidden_states):
        return self.layers(hidden_states)


class UnresolvedInput(LoopBlocks):
    def forward(self, payload):
        for block in self.layers:
            payload = block(payload, use_cache=False, output_attentions=False)[0]
        return payload


def _analyze(model: nn.Module):
    classifications = {
        name: SimpleNamespace(kind="backbone") for name, _ in model.named_modules()
    }
    return StaticFlowAnalyzer(model.eval(), "", classifications).analyze()


def _member_event(flow):
    return next(event for event in flow.events if event.metadata.get("native_container") == "layers")


def test_unreviewed_member_source_does_not_claim_common_tuple_output() -> None:
    flow = _analyze(LoopBlocks([_block(), _block()]))
    assert not any(
        event.metadata.get("contract") == "reviewed_native_module_list_member.v1"
        for event in flow.events
    )
    assert flow.outputs.spec is None or flow.outputs.spec.shape is None

def test_direct_modulelist_call_does_not_receive_member_contract() -> None:
    flow = _analyze(DirectContainerCall([_block()]))

    assert not any(event.metadata.get("native_container") == "layers" for event in flow.events)
    assert not any(event.metadata.get("contract") == "reviewed_native_module_list_member.v1" for event in flow.events)


def test_empty_modulelist_loop_does_not_claim_member_outputs() -> None:
    flow = _analyze(LoopBlocks([]))

    assert not any(event.metadata.get("native_container") == "layers" for event in flow.events)
    assert not any("output_port_specs" in event.metadata for event in flow.events)


def test_mixed_native_and_unreviewed_members_decline_the_common_contract() -> None:
    flow = _analyze(LoopBlocks([_block(), nn.Identity()]))

    assert not any(event.metadata.get("native_container") == "layers" for event in flow.events)
    assert flow.outputs.spec is None or flow.outputs.spec.shape is None


def test_member_width_mismatch_declines_the_common_contract() -> None:
    flow = _analyze(LoopBlocks([_block(16), _block(32)]))

    assert not any(event.metadata.get("native_container") == "layers" for event in flow.events)
    assert flow.outputs.spec is None or flow.outputs.spec.shape is None


def test_container_or_member_hooks_decline_the_common_contract() -> None:
    hooked = LoopBlocks([_block(), _block()])
    hooked.layers.register_forward_hook(lambda *_args: None)
    container_flow = _analyze(hooked)
    assert not any(event.metadata.get("native_container") == "layers" for event in container_flow.events)

    hooked_member = LoopBlocks([_block(), _block()])
    hooked_member.layers[1].register_forward_hook(lambda *_args: None)
    member_flow = _analyze(hooked_member)
    assert not any(event.metadata.get("native_container") == "layers" for event in member_flow.events)


def test_unknown_input_shape_declines_the_common_contract() -> None:
    flow = _analyze(UnresolvedInput([_block()]))

    assert not any(event.metadata.get("native_container") == "layers" for event in flow.events)
    assert flow.outputs.spec is None or flow.outputs.spec.shape is None
