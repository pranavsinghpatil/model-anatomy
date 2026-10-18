"""Reusable QKV attention refinement driven by discovered projection topology."""

from __future__ import annotations

import hashlib
import inspect
import re
from typing import TYPE_CHECKING

from .ir import Graph, Node, TensorSpec

if TYPE_CHECKING:
    from .hierarchy import HierarchyBuilder


def build_attention(builder: HierarchyBuilder, path: str) -> Graph:
    raw = builder.raw
    module = builder.model.get_submodule(path)
    children = {p.rsplit(".", 1)[-1].lower(): p for p in raw["modules"][path]["children"]}
    attributes = raw["modules"][path]["attributes"]
    hidden = builder.inputs.spec("hidden_states", path)
    sequence = hidden.shape[1] if hidden.shape else "S"
    cross = builder.analyzer.classifications[path].kind == "cross_attention"
    key_sequence = "S" if cross else sequence
    heads = next(
        (attributes[k] for k in ("num_heads", "num_attention_heads", "n_heads") if k in attributes),
        getattr(getattr(module, "config", builder.model.config), "num_attention_heads", None)
        or getattr(getattr(module, "config", builder.model.config), "num_heads", None),
    )
    head_dim = attributes.get("head_dim") or attributes.get("key_value_proj_dim")
    fused = "c_attn" in children
    projection_sets = [("q_proj", "k_proj", "v_proj"), ("query", "key", "value"), ("q", "k", "v")]
    names = next((group for group in projection_sets if all(name in children for name in group)), None)
    output_path = next(
        (children[name] for name in ("o_proj", "c_proj", "o", "out_proj", "dense") if name in children), None
    )
    if heads is None or (not fused and names is None):
        return builder.source_graph(path)
    if fused:
        packed_path = children["c_attn"]
        width = builder.model.get_submodule(packed_path).nf
        query_width = width // 3
        kv_width = query_width
        projection_specs = [TensorSpec(["B", sequence, width], builder.inputs.activation_dtype, "packed_qkv")]
    else:
        query_width = builder.model.get_submodule(children[names[0]]).out_features
        kv_width = builder.model.get_submodule(children[names[1]]).out_features
        value_width = builder.model.get_submodule(children[names[2]]).out_features
        if kv_width != value_width:
            return builder.source_graph(path)
        projection_specs = [
            TensorSpec(["B", sequence, query_width], builder.inputs.activation_dtype, "q"),
            TensorSpec(["B", key_sequence, kv_width], builder.inputs.activation_dtype, "k"),
            TensorSpec(["B", key_sequence, kv_width], builder.inputs.activation_dtype, "v"),
        ]
    head_dim = head_dim or query_width // heads
    if query_width != heads * head_dim or kv_width % head_dim:
        return builder.source_graph(path)
    kv_heads = kv_width // head_dim
    if heads % kv_heads:
        return builder.source_graph(path)
    if hidden.shape is None:
        hidden = TensorSpec(["B", sequence, query_width], builder.inputs.activation_dtype, "hidden_states")
    output = TensorSpec(["B", sequence, query_width], builder.inputs.activation_dtype, "context")
    try:
        source = inspect.getsource(type(module).forward)
        kernel = getattr(inspect.getmodule(type(module)), "eager_attention_forward", None)
        kernel_source = inspect.getsource(kernel) if kernel else source
    except (OSError, TypeError):
        return builder.source_graph(path)
    if "softmax" not in kernel_source or not any(
        name in kernel_source for name in ("matmul", "bmm", "einsum")
    ):
        return builder.source_graph(path)
    graph = Graph(
        builder.graph_id(path),
        [],
        [],
        {
            "module_path": path,
            "topology": "QKV attention structural pattern",
            "runtime_verified": False,
            "query_heads": heads,
            "kv_heads": kv_heads,
            "head_dimension": head_dim,
            "projection_layout": "fused" if fused else "separate",
            "kernel_source": {
                "qualified_name": f"{kernel.__module__}.{kernel.__name__}"
                if kernel
                else f"{type(module).__module__}.{type(module).__name__}.forward",
                "sha256": hashlib.sha256(kernel_source.encode()).hexdigest(),
            },
        },
    )
    entry = Node("input", "Input", "input", outputs=[hidden])
    graph.nodes.append(entry)
    if fused:
        projection = builder.module_node(packed_path, "qkv", build_child=False)
        projection.label = "QKV Projection"
        projection.outputs = projection_specs
        projection.metadata.update(
            in_features=query_width,
            out_features=3 * query_width,
            weight_layout="input_features, output_features",
        )
    else:
        projection_graph = Graph(
            builder.graph_id(path, graph.id + "_qkv"),
            [],
            [],
            {"module_path": path, "runtime_verified": False},
        )
        q_input = Node("input", "Query input", "input", outputs=[hidden])
        projection_graph.nodes.append(q_input)
        kv_input = q_input
        if cross:
            kv_input = Node(
                "encoder",
                "Encoder context",
                "input",
                outputs=[
                    TensorSpec(
                        ["B", "S", hidden.shape[-1]], builder.inputs.activation_dtype, "encoder_hidden_states"
                    )
                ],
            )
            projection_graph.nodes.append(kv_input)
        projection_output = Node("output", "Q / K / V", "output", outputs=projection_specs)
        for index, name in enumerate(names):
            p = builder.module_node(children[name], ["q", "k", "v"][index], build_child=False)
            p.label, p.outputs = (
                ["Q Projection", "K Projection", "V Projection"][index],
                [projection_specs[index]],
            )
            projection_graph.nodes.append(p)
            builder.connect(projection_graph, q_input if index == 0 else kv_input, p)
            builder.connect(
                projection_graph,
                p,
                projection_output,
                projection_specs[index],
                source_port=["q", "k", "v"][index],
            )
        projection_graph.nodes.append(projection_output)
        projection = Node(
            "qkv",
            "Q / K / V Projections",
            "projection_group",
            child_graph=projection_graph,
            outputs=projection_specs,
            source_modules=[children[name] for name in names],
            evidence=[builder.evidence(path, "separate_qkv_projection_pattern")],
            confidence=0.99,
        )
    graph.nodes.append(projection)
    builder.connect(graph, entry, projection)
    if cross:
        encoder = Node(
            "encoder",
            "Encoder context",
            "input",
            outputs=[
                TensorSpec(
                    ["B", "S", hidden.shape[-1]], builder.inputs.activation_dtype, "encoder_hidden_states"
                )
            ],
        )
        graph.nodes.append(encoder)
        builder.connect(graph, encoder, projection, target_port="key_value_states")
    core = Graph(
        builder.graph_id(
            path,
            graph.id + "_core",
        ),
        [],
        [],
        {
            "module_path": path,
            "runtime_verified": False,
            "shape_basis": "projection shapes + config heads",
            "mask_role": "cross_attention"
            if cross
            else "causal"
            if attributes.get("is_causal") or attributes.get("is_decoder")
            else "attention mask / bias",
        },
    )
    q_head = TensorSpec(["B", heads, sequence, head_dim], builder.inputs.activation_dtype, "q")
    kv_head = TensorSpec(["B", kv_heads, key_sequence, head_dim], builder.inputs.activation_dtype, "kv")
    scores = TensorSpec(["B", heads, sequence, key_sequence], builder.inputs.activation_dtype, "scores")

    def n(id: str, label: str, kind: str, outputs: list[TensorSpec], metadata: dict | None = None) -> Node:
        node = Node(
            id,
            label,
            kind,
            outputs=outputs,
            metadata=metadata or {},
            evidence=[builder.evidence(path, f"attention_pattern:{kind}")],
            confidence=0.95,
        )
        core.nodes.append(node)
        return node

    core_input = n("input", "Packed QKV" if fused else "Q / K / V", "input", projection_specs)
    split = n(
        "split",
        "Split QKV and reshape heads" if fused else "Reshape Q / K / V heads",
        "split_heads",
        [q_head, kv_head, kv_head],
    )
    for index, spec in enumerate(projection_specs):
        builder.connect(
            core, core_input, split, spec, source_port=["q", "k", "v"][index] if not fused else "qkv"
        )
    q = n("q", "Q", "query", [q_head])
    k = n("k", "K", "key", [kv_head])
    v = n("v", "V", "value", [kv_head])
    for node, spec, port in [(q, q_head, "q"), (k, kv_head, "k"), (v, kv_head, "v")]:
        builder.connect(core, split, node, spec, source_port=port)
    if "apply_rotary" in source:
        q_rot = n("rope_q", "Rotary position: Q", "rotary_position", [q_head])
        k_rot = n("rope_k", "Rotary position: K", "rotary_position", [kv_head])
        builder.connect(core, q, q_rot)
        builder.connect(core, k, k_rot)
        q, k = q_rot, k_rot
    if kv_heads != heads:
        expanded = TensorSpec(
            ["B", heads, key_sequence, head_dim], builder.inputs.activation_dtype, "expanded_kv"
        )
        k_repeat = n("repeat_kv_k", "Repeat K heads", "repeat_kv", [expanded], {"groups": heads // kv_heads})
        v_repeat = n("repeat_kv_v", "Repeat V heads", "repeat_kv", [expanded], {"groups": heads // kv_heads})
        builder.connect(core, k, k_repeat)
        builder.connect(core, v, v_repeat)
        k, v = k_repeat, v_repeat
    dot = n("scores", "Q @ K transpose", "matmul", [scores])
    builder.connect(core, q, dot, target_port="q")
    builder.connect(core, k, dot, target_port="k")
    scaling = attributes.get("scaling")
    explicit = re.search(r"scaling\s*=\s*(\d+(?:\.\d+)?)", source)
    if explicit:
        scaling = float(explicit[1])
    if scaling is None and ("** -0.5" in kernel_source or "sqrt" in kernel_source):
        scaling = head_dim**-0.5
    scale = n(
        "scale", "Scale attention scores", "scale", [scores], {"factor": scaling, "head_dimension": head_dim}
    )
    builder.connect(core, dot, scale)
    mask = n(
        "mask",
        "Attention mask / position bias",
        "attention_mask",
        [scores],
        {"mask_role": core.metadata["mask_role"]},
    )
    builder.connect(core, scale, mask)
    relative = next((p for name, p in children.items() if "relative_attention_bias" in name), None)
    if relative:
        bias = builder.module_node(relative, "relative_bias", build_child=False)
        bias.outputs = [
            TensorSpec(
                [1, heads, sequence, key_sequence], builder.inputs.activation_dtype, "relative_position_bias"
            )
        ]
        core.nodes.append(bias)
        builder.connect(core, bias, mask, target_port="position_bias")
    softmax = n("softmax", "Softmax", "softmax", [scores], {"axis": -1})
    builder.connect(core, mask, softmax)
    last = softmax
    attn_dropout = next(
        (p for name, p in children.items() if name in {"attn_dropout", "attention_dropout", "dropout"}), None
    )
    if attn_dropout:
        last = builder.module_node(attn_dropout, "dropout", build_child=False)
        last.outputs = [scores]
        core.nodes.append(last)
        builder.connect(core, softmax, last)
    aggregate = n("aggregate", "Attention probabilities @ V", "matmul", [q_head])
    builder.connect(core, last, aggregate, target_port="probabilities")
    builder.connect(core, v, aggregate, target_port="v")
    transposed = TensorSpec(
        ["B", sequence, heads, head_dim], builder.inputs.activation_dtype, "transposed_context"
    )
    transpose = n("transpose", "Transpose sequence and head axes", "transpose", [transposed])
    merge = n("merge", "Merge heads", "merge_heads", [output])
    core_output = n("output", "Context", "output", [output])
    builder.connect(core, aggregate, transpose)
    builder.connect(core, transpose, merge)
    builder.connect(core, merge, core_output)
    core_node = Node(
        "core",
        "Attention Core",
        "attention_core",
        child_graph=core,
        outputs=[output],
        evidence=[builder.evidence(path, "qkv_attention_core_pattern")],
        confidence=0.95,
    )
    graph.nodes.append(core_node)
    for index, spec in enumerate(projection_specs):
        builder.connect(
            graph, projection, core_node, spec, source_port=["q", "k", "v"][index] if not fused else "qkv"
        )
    current = core_node
    if output_path is not None:
        final_projection = builder.module_node(output_path, "projection", build_child=False)
        final_projection.label, final_projection.outputs = "Output Projection", [hidden]
        graph.nodes.append(final_projection)
        builder.connect(graph, core_node, final_projection)
        current = final_projection
    if "resid_dropout" in children:
        previous = current
        current = builder.module_node(children["resid_dropout"], "dropout", build_child=False)
        current.outputs = [hidden]
        graph.nodes.append(current)
        builder.connect(graph, previous, current)
    graph_output = Node("output", "Output", "output", outputs=[hidden])
    graph.nodes.append(graph_output)
    builder.connect(graph, current, graph_output)
    return graph
