"""Native ViT semantic-flow regressions using metadata-only reconstruction."""

from __future__ import annotations

from test_engine import FixtureClient, checkpoint_for

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

    output_modules = [
        node
        for node in nodes
        if node.module_path
        and node.module_path.rsplit(".", 1)[-1] in {"output", "attention_output"}
        and node.module_type in {"ViTOutput", "ViTSelfOutput"}
    ]
    assert {node.module_type for node in output_modules} == {"ViTOutput", "ViTSelfOutput"}
    assert all(node.kind == "projection_block" for node in output_modules)
    assert all(
        any(
            "native ViT" in reason and "forward source" in reason
            for reason in node.metadata["classification"]["evidence"]
        )
        for node in output_modules
    )

    vit_output = next(node for node in output_modules if node.module_type == "ViTOutput")
    assert ["B", 5, 48] in [tensor.shape for tensor in vit_output.inputs]
    assert output_shape(vit_output) == ["B", 5, 24]
    output_graph = vit_output.child_graph
    assert output_graph is not None
    residual = next(n for n in output_graph.nodes if n.kind == "residual")
    incoming = [e for e in output_graph.edges if e.target == residual.id and e.relation == "dataflow"]
    by_id = {n.id: n for n in output_graph.nodes}
    assert len(incoming) == 2
    assert {by_id[e.source].kind for e in incoming} == {"dropout", "input"}
    skip = next(e for e in incoming if e.target_port == "skip")
    assert by_id[skip.source].metadata["argument_name"] == "input_tensor"
    assert all(e.tensor.shape == ["B", 5, 24] and e.evidence for e in incoming)
    assert output_shape(residual) == ["B", 5, 24]
    assert any(e.source == residual.id and by_id[e.target].kind == "output" for e in output_graph.edges)
    self_output = next(n for n in output_modules if n.module_type == "ViTSelfOutput")
    assert self_output.child_graph is not None
    assert not any(n.kind == "residual" for n in self_output.child_graph.nodes)

    unknown_modules = ir.raw_structure["semantic_diagnostics"]["unknown_components"]
    assert all(node.module_path not in unknown_modules for node in output_modules)


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
