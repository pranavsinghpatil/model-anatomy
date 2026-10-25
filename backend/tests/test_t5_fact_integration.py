"""Opt-in pinned acceptance for the native T5 mask dependency chain."""

import os

import pytest

from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests")
@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("google-t5/t5-small", "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"),
        ("google-t5/t5-base", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"),
    ],
)
def test_pinned_t5_mask_facts_reach_the_decoder(model_id, revision):
    ir = reconstruct(model_id, revision=revision)
    assert ir.model_id == model_id
    assert ir.metadata["revision"] == revision
    assert ir.metadata["source_config"]["model_type"] == "t5"
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    analysis = ir.metadata["analysis_report"]
    assert analysis["forward_executed"] is False
    for metric in ("semantic_coverage", "module_semantic_coverage", "checkpoint_parameter_coverage"):
        assert analysis[metric] == 1
    assert analysis["tensor_shape_coverage"] > 0.9035087719298246

    decoder = next(graph for graph in ir.root_graph.walk() if graph.id == "decoder")
    required = {
        "torch.full",
        "torch.triu",
        "torch.Tensor.expand",
        "torch.Tensor.clone",
        "torch.Tensor.masked_fill",
    }
    operations = {node.metadata.get("callable"): node for node in decoder.nodes}
    assert required <= operations.keys()
    for name in required:
        node = operations[name]
        assert node.outputs and all(output.shape is not None and output.dtype == "F32" for output in node.outputs)
    for name in ("torch.Tensor.expand", "torch.Tensor.clone", "torch.Tensor.masked_fill"):
        assert operations[name].outputs[0].shape == ["B", 1, "T", "T"]

    # The padding update must feed decoder execution, rather than becoming an
    # orphan node while the original pre-update clone is returned.
    reachable = {operations["torch.Tensor.masked_fill"].id}
    while True:
        expanded = reachable | {
            edge.target for edge in decoder.edges if edge.relation == "dataflow" and edge.source in reachable
        }
        if expanded == reachable:
            break
        reachable = expanded
    assert any(node.id in reachable and node.module_path == "decoder.block" for node in decoder.nodes)
    ir.root_graph.validate()
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
