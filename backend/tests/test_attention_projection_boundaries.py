"""Boundary checks for attention projection refinement; forwards are never invoked."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model_anatomy.src.attention import build_attention
from model_anatomy.src.ir import Edge, Evidence, Graph, Node, TensorSpec


class Projection(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features


class WrappedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)


class PackedProjection(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf


class Attention(nn.Module):
    def __init__(self, *, heads=2, head_dim=None, projection="separate", width=4):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=heads, num_heads=None)
        if projection == "fused":
            self.c_attn = PackedProjection(width)
        else:
            self.q_proj = Projection(width)
            self.k_proj = Projection(width)
            self.v_proj = Projection(width)
        if head_dim is not None:
            self.head_dim = head_dim

    def forward(self, hidden_states):
        # Source inspection recognizes this as the ordinary attention kernel.
        return torch.softmax(torch.matmul(hidden_states, hidden_states), dim=-1)


class RootModel(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, num_heads=None)
        self.attn = attention


class Inputs:
    activation_dtype = "F32"

    def spec(self, name, path):
        return TensorSpec(["B", "S", 4], self.activation_dtype, name)


class Builder:
    def __init__(self, attention):
        self.model = RootModel(attention)
        names = list(dict(attention.named_children()))
        self.raw = {
            "modules": {
                "attn": {
                    "children": [f"attn.{name}" for name in names],
                    "attributes": {"num_heads": attention.config.num_attention_heads},
                }
            }
        }
        if hasattr(attention, "head_dim"):
            self.raw["modules"]["attn"]["attributes"]["head_dim"] = attention.head_dim
        self.inputs = Inputs()
        self.analyzer = SimpleNamespace(
            classifications={"attn": SimpleNamespace(kind="self_attention")}, adapter=None
        )
        self.source_graph_calls = 0

    def source_graph(self, path):
        self.source_graph_calls += 1
        return Graph("source", [], [], {"module_path": path})

    def graph_id(self, path, suggested=None):
        return suggested or path

    def evidence(self, path, rule, kind="reconstructed"):
        return Evidence("test", kind, rule)

    def module_node(self, path, node_id, repeat=None, build_child=True):
        return Node(node_id, path, "module", module_path=path)

    def connect(self, graph, source, target, tensor=None, source_port="output", target_port="input"):
        graph.edges.append(Edge(source.id, target.id, tensor, source_port, target_port))


def assert_source_fallback(builder):
    graph = build_attention(builder, "attn")
    assert graph.metadata["module_path"] == "attn"
    assert builder.source_graph_calls == 1


def test_wrapped_qkv_without_out_features_uses_source_graph():
    attention = Attention()
    attention.q_proj = WrappedProjection()
    attention.k_proj = WrappedProjection()
    attention.v_proj = WrappedProjection()
    builder = Builder(attention)

    assert_source_fallback(builder)


@pytest.mark.parametrize("heads", [0, -1, True])
def test_invalid_head_counts_use_source_graph_without_division(heads):
    builder = Builder(Attention(heads=heads))

    assert_source_fallback(builder)


@pytest.mark.parametrize("head_dim", [0, -1, True])
def test_invalid_explicit_head_dimensions_use_source_graph(head_dim):
    builder = Builder(Attention(head_dim=head_dim))

    assert_source_fallback(builder)


@pytest.mark.parametrize("width", [0, -3, 10])
def test_invalid_fused_projection_width_uses_source_graph(width):
    builder = Builder(Attention(projection="fused", width=width))

    assert_source_fallback(builder)


@pytest.mark.parametrize("width", [0, -1, True, None])
def test_invalid_separate_projection_width_uses_source_graph(width):
    attention = Attention()
    attention.q_proj = Projection(width)
    builder = Builder(attention)

    assert_source_fallback(builder)


def test_valid_standard_width_builds_attention_graph_without_fallback():
    builder = Builder(Attention(heads=2, head_dim=2))

    graph = build_attention(builder, "attn")

    assert builder.source_graph_calls == 0
    assert graph.metadata["query_heads"] == 2
    assert graph.metadata["head_dimension"] == 2
    assert graph.metadata["projection_layout"] == "separate"
