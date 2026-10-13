"""Native ViT semantic-flow regressions using metadata-only reconstruction."""

from __future__ import annotations

from test_engine import FixtureClient, checkpoint_for

from model_anatomy.service.jobs import ir_summary
from model_anatomy.service.views import node_view
from model_anatomy.src.pipeline import reconstruct
from model_anatomy.src.registry import instantiate_native, load_native_config


def vit_config(image_size=32):
    return {
        "model_type": "vit",
        "architectures": ["ViTForImageClassification"],
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_attention_heads": 3,
        "num_hidden_layers": 1,
        "image_size": image_size,
        "patch_size": 16,
        "num_channels": 3,
        "num_labels": 10,
    }


def reconstruct_vit(config):
    model, _ = instantiate_native(load_native_config(config))
    return reconstruct("example/vision", client=FixtureClient(checkpoint_for(model), config))


def all_nodes(ir):
    return [node for graph in ir.root_graph.walk() for node in graph.nodes]


def module_node(nodes, path):
    return next(node for node in nodes if node.module_path == path)


def output_shape(node):
    assert node.outputs
    return node.outputs[0].shape


def source_operation(nodes, method):
    return next(node for node in nodes if node.metadata.get("method") == method and node.module_path is None)


def test_native_vit_static_semantics_and_source_flow():
    ir = reconstruct_vit(vit_config())
    nodes = all_nodes(ir)

    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    assert ir.metadata["inputs"]["input_specs"]["pixel_values"]["shape"] == ["B", 3, 32, 32]

    patch_projection = module_node(nodes, "vit.embeddings.patch_embeddings.projection")
    assert patch_projection.kind == "convolution"
    assert output_shape(patch_projection) == ["B", 24, 2, 2]

    expected_operations = {
        "flatten": ["B", 24, 4],
        "transpose": ["B", 4, 24],
        "expand": ["B", 1, 24],
    }
    for method, shape in expected_operations.items():
        assert output_shape(source_operation(nodes, method)) == shape

    concatenation = next(
        node for node in nodes if node.kind == "concatenation" and node.metadata.get("dimension") == 1
    )
    assert output_shape(concatenation) == ["B", 5, 24]

    cls_slice = next(
        node for node in nodes if node.kind == "selection" and node.metadata.get("indices") == [None, 0, None]
    )
    assert output_shape(cls_slice) == ["B", 24]

    embeddings_graph = next(
        graph for graph in ir.root_graph.walk() if graph.metadata.get("module_path") == "vit.embeddings"
    )
    position_add = next(
        node for node in embeddings_graph.nodes if node.kind == "add" and output_shape(node) == ["B", 5, 24]
    )
    assert output_shape(position_add) == ["B", 5, 24]

    classifier = module_node(nodes, "classifier")
    assert classifier.kind == "classifier"
    assert output_shape(classifier) == ["B", 10]

    # The pinned native runtime places both residual additions in ViTLayer.
    layer = module_node(nodes, "vit.layers.0")
    layer_graph = layer.child_graph
    assert layer_graph is not None
    by_id = {node.id: node for node in layer_graph.nodes}
    residuals = [node for node in layer_graph.nodes if node.kind == "residual"]
    assert len(residuals) == 2
    for index, residual in enumerate(residuals):
        incoming = [
            edge for edge in layer_graph.edges if edge.target == residual.id and edge.relation == "dataflow"
        ]
        assert len(incoming) == 2
        skip = next(edge for edge in incoming if edge.target_port == "skip")
        assert by_id[skip.source].kind == ("input" if index == 0 else "residual")
        if index:
            assert skip.source == residuals[0].id
        branch = next(edge for edge in incoming if edge.target_port == "left")
        assert by_id[branch.source].kind == "dropout"
        assert all(edge.tensor.shape == ["B", 5, 24] and edge.evidence for edge in incoming)
        assert output_shape(residual) == ["B", 5, 24]
        assert residual.evidence
    assert any(
        edge.source == residuals[-1].id and by_id[edge.target].kind == "output" for edge in layer_graph.edges
    )

    expansion = module_node(nodes, "vit.layers.0.mlp.fc1")
    contraction = module_node(nodes, "vit.layers.0.mlp.fc2")
    assert output_shape(expansion) == ["B", 5, 48]
    assert output_shape(contraction) == ["B", 5, 24]
    attention_projection = module_node(nodes, "vit.layers.0.attention.o_proj")
    assert output_shape(attention_projection) == ["B", 5, 24]
    assert all(node.evidence for node in (expansion, contraction, attention_projection))


def test_native_vit_nonsquare_image_size_preserves_patch_grid_shape():
    ir = reconstruct_vit(vit_config([32, 48]))
    nodes = all_nodes(ir)

    assert ir.metadata["inputs"]["input_specs"]["pixel_values"]["shape"] == ["B", 3, 32, 48]
    patch_projection = module_node(nodes, "vit.embeddings.patch_embeddings.projection")
    assert output_shape(patch_projection) == ["B", 24, 2, 3]
    assert output_shape(source_operation(nodes, "flatten")) == ["B", 24, 6]
    assert output_shape(source_operation(nodes, "transpose")) == ["B", 6, 24]
    concatenation = next(node for node in nodes if node.kind == "concatenation")
    assert output_shape(concatenation) == ["B", 7, 24]


def test_cls_and_position_parameter_nodes_expose_only_their_checkpoint_tensors():
    ir = reconstruct_vit(vit_config())
    graph = next(g for g in ir.root_graph.walk() if g.metadata.get("module_path") == "vit.embeddings")
    for name, count in (("vit.embeddings.cls_token", 24), ("vit.embeddings.position_embeddings", 120)):
        node = next(n for n in graph.nodes if n.metadata.get("parameter_refs") == [name])
        detail = node_view(ir, graph.id, node.id)
        assert node.parameter_count == count
        assert detail["tensor_count"] == 1
        assert detail["tensors"][0]["name"] == name
    for node in graph.nodes:
        if node.kind in {"reshape", "concatenation", "add"}:
            assert node.evidence
            assert all(
                e.details["source_fingerprint"] and e.details["runtime_verified"] is False
                for e in node.evidence
            )


def test_native_vit_qkv_exports_preserve_projection_tensor_specs():
    ir = reconstruct_vit(vit_config())

    qkv_graphs = [graph for graph in ir.root_graph.walk() if graph.id.endswith("_qkv")]
    assert qkv_graphs
    for graph in qkv_graphs:
        by_id = {node.id: node for node in graph.nodes}
        exports = [edge for edge in graph.edges if edge.target == "output"]
        assert len(exports) == 3
        assert {edge.source_port for edge in exports} == {"q", "k", "v"}
        for edge in exports:
            projection = by_id[edge.source]
            assert projection.outputs
            spec = projection.outputs[0]
            assert edge.tensor.shape == spec.shape
            assert edge.tensor.dtype == spec.dtype

    report = ir.metadata["analysis_report"]
    assert ir_summary(ir.to_dict())["result_status"] == "completed"
    assert report["tensor_shape_coverage"] == 1
    assert report["warnings"] == []
