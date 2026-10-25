"""Opt-in pinned acceptance for native T5 encoder/decoder module architecture."""

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
def test_pinned_t5_registered_modules_and_checkpoint_binding(model_id, revision):
    ir = reconstruct(model_id, revision=revision)
    assert ir.model_id == model_id
    assert ir.metadata["revision"] == revision
    assert ir.metadata["source_config"]["model_type"] == "t5"
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    analysis = ir.metadata["analysis_report"]
    assert analysis["forward_executed"] is False
    assert analysis["mode"] == "static_module_architecture"
    assert analysis["checkpoint_parameter_coverage"] == 1
    assert analysis["completion"]["checks"]["module_inventory"] is True
    assert analysis["completion"]["checks"]["parameter_mapping"] is True
    nodes = [node for graph in ir.root_graph.walk() for node in graph.nodes]
    paths = {node.module_path for node in nodes}
    assert {"encoder", "decoder", "encoder.block", "decoder.block"} <= paths
    for branch in ("encoder", "decoder"):
        container = next(node for node in nodes if node.module_path == f"{branch}.block")
        instances = container.metadata["instance_paths"]
        assert len(instances) == container.metadata["instance_count"]
        assert container.child_graph is not None
        assert {node.module_path for node in container.child_graph.nodes if node.module_path} == set(instances)
    assert all(node.module_path is not None or node.kind in {"input", "output"} for node in nodes)
    ir.root_graph.validate()
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
