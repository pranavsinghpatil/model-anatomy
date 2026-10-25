import pytest

from model_anatomy.service import views
from model_anatomy.service.errors import ServiceError
from model_anatomy.src.ir import Edge, Evidence, Graph, ModelIR, Node
from model_anatomy.src.version import ENGINE_VERSION


def contract_report():
    """Create independent HTTP test data; this is not evidence about a real model.

    The repository identity exercises the API's existing short-name normalization.
    Shapes, counts and topology below are deliberately tiny synthetic fixtures.
    No extraction, model-library import, network access or weight file is needed.
    """
    revision = "a" * 40
    evidence = [Evidence("contract_fixture", "direct", "test-only")]
    norm = Node("norm", "LayerNorm", "normalization", "block.norm",
                "torch.nn.LayerNorm", parameter_count=8, evidence=evidence)
    block = Node("block", "Block", "module", "block", "ContractBlock",
                 parameter_count=8, evidence=evidence,
                 child_graph=Graph("block_graph", [norm], []))
    embedding = Node("embedding", "Embedding", "embedding", "embedding",
                     "torch.nn.Embedding", parameter_count=32, evidence=evidence)
    root = Graph("root", [embedding, block], [
        Edge("embedding", "block", None, evidence=evidence, relation="dependency"),
    ])
    shapes = {"embedding.weight": [8, 4], "block.norm.weight": [4], "block.norm.bias": [4]}
    tensors, parameters = {}, {}
    for name, shape in shapes.items():
        count = 32 if len(shape) == 2 else 4
        tensors[name] = {
            "shape": shape, "dtype": "F32", "parameter_count": count,
            "artifact": f"https://huggingface.co/openai-community/gpt2/resolve/{revision}/model.safetensors",
            "parameter_refs": [name],
        }
        parameters[name] = {"canonical_parameter": name, "parameter_count": count,
                            "checkpoint_matched": True, "tensor_names": [name]}
    return ModelIR(
        model_id="openai-community/gpt2", architecture="ContractModel", root_graph=root,
        tensors=tensors, evidence=evidence,
        raw_structure={"modules": {"": {"type": "ContractModel", "attributes": {}}},
                       "parameters": parameters, "buffers": {}},
        metadata={
            "test_fixture": True, "revision": revision, "engine_version": ENGINE_VERSION,
            "unique_parameter_count": 40, "logical_parameter_count": 40,
            "checkpoint_tensor_element_count": 40,
            "dimensions": {"hidden": 4, "layers": 1, "vocabulary": 8},
            "inputs": {"input_specs": {"input_ids": {"shape": None, "dtype": "I64"}}},
            "artifacts": {"weight_payload_bytes_read": 0, "repository_files": ["config.json", "model.safetensors"]},
            "limitations": ["Synthetic service contract fixture; not parser acceptance evidence"],
            "analysis_report": {
                "mode": "static_module_architecture", "forward_executed": False,
                "runtime_trace_available": False, "checkpoint_parameter_coverage": 1,
                "semantic_coverage": 1, "module_semantic_coverage": 1,
                "unknown_nodes": 0, "unmapped_checkpoint_tensors": [], "warnings": [],
                "completion": {"version": 3, "status": "complete", "scope": "Service contract fixture",
                               "checks": {"module_inventory": True, "parameter_mapping": True,
                                          "native_architecture": True, "module_topology": True}},
            },
        },
    )


def test_views_do_not_mutate_stored_report():
    ir = contract_report()
    before = ir.to_json()
    graph = views.graph_view(ir, ir.root_graph.id)
    views.node_view(ir, ir.root_graph.id, graph["nodes"][0]["id"])
    assert ir.to_json() == before


def test_artifact_checkpoint_availability_uses_full_report_not_page():
    report = contract_report()
    assert report.tensors
    page = views.artifact_view(report, offset=len(report.tensors), limit=50)
    assert page["tensors"] == []
    assert page["checkpoint_available"] is True


def test_artifact_view_preserves_explicit_checkpoint_unavailable():
    report = contract_report()
    report.metadata["inspection"] = {"checkpoint_available": False}
    assert views.artifact_view(report, offset=len(report.tensors))["checkpoint_available"] is False


@pytest.mark.parametrize("data", [b"not-json", b"{}", b'{"schema_version":"invalid"}'])
def test_invalid_stored_report_fails_honestly(data):
    with pytest.raises(ServiceError) as exc:
        views.load_ir(data)
    assert exc.value.code == "invalid_artifact"
