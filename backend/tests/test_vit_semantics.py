"""Native ViT architecture regressions using metadata-only reconstruction."""

from test_engine import FixtureClient, checkpoint_for

from model_anatomy.service.views import node_view
from model_anatomy.src.pipeline import reconstruct
from model_anatomy.src.registry import instantiate_native, load_native_config


def vit_config(image_size=32):
    return {
        "model_type": "vit", "architectures": ["ViTForImageClassification"],
        "hidden_size": 24, "intermediate_size": 48, "num_attention_heads": 3,
        "num_hidden_layers": 1, "image_size": image_size, "patch_size": 16,
        "num_channels": 3, "num_labels": 10,
    }


def reconstruct_vit(config):
    model, _ = instantiate_native(load_native_config(config))
    return reconstruct("example/vision", client=FixtureClient(checkpoint_for(model), config))


def all_nodes(ir):
    return [node for graph in ir.root_graph.walk() for node in graph.nodes]


def module_node(nodes, path):
    return next(node for node in nodes if node.module_path == path)


def test_native_vit_static_semantics_and_source_flow():
    ir = reconstruct_vit(vit_config())
    nodes = all_nodes(ir)
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    patch = module_node(nodes, "vit.embeddings.patch_embeddings.projection")
    assert patch.kind == "convolution" and patch.child_graph is None
    assert module_node(nodes, "classifier").kind == "classifier"
    assert all(n.module_path in ir.raw_structure["modules"] or n.kind == "input" for n in nodes)
    assert all(e.evidence for g in ir.root_graph.walk() for e in g.edges)


def test_native_vit_nonsquare_image_size_preserves_checkpoint_shapes():
    ir = reconstruct_vit(vit_config([32, 48]))
    assert ir.metadata["source_config"]["image_size"] == [32, 48]
    assert ir.tensors["vit.embeddings.position_embeddings"]["shape"] == [1, 7, 24]
    assert ir.tensors["vit.embeddings.patch_embeddings.projection.weight"]["shape"] == [24, 3, 16, 16]


def test_embedding_module_inspection_preserves_direct_parameter_tensors():
    ir = reconstruct_vit(vit_config())
    graph = next(g for g in ir.root_graph.walk() if any(n.module_path == "vit.embeddings" for n in g.nodes))
    embeddings = next(n for n in graph.nodes if n.module_path == "vit.embeddings")
    detail = node_view(ir, graph.id, embeddings.id)
    names = {tensor["name"] for tensor in detail["tensors"]}
    assert {"vit.embeddings.cls_token", "vit.embeddings.position_embeddings"} <= names
    assert all(name.startswith("vit.embeddings.") for name in names)


def test_native_vit_keeps_actual_projection_modules():
    ir = reconstruct_vit(vit_config())
    nodes = all_nodes(ir)
    for name in ("q_proj", "k_proj", "v_proj"):
        node = module_node(nodes, "vit.layers.0.attention." + name)
        assert node.kind == "projection" and node.child_graph is None
        assert node.parameter_count == 24 * 24 + 24
    assert ir.metadata["analysis_report"]["module_inventory_coverage"] == 1
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
