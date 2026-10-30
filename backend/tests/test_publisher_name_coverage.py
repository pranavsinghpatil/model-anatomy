"""The pinned publisher state-key grammar must match its saved audit evidence."""

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from model_anatomy.src.artifacts import ModelReference
from model_anatomy.src.ir import Graph, ModelIR
from model_anatomy.src.publisher_name_coverage import (
    audit_name_coverage,
    converted_name,
    expected_state_keys,
)
from model_anatomy.src.publisher_topology import (
    MODEL_ID,
    REQUIRED_FILES,
    REVISION,
    build_reviewed_topology,
)

ROOT = Path(__file__).parents[2]
CONFIG = json.loads((Path(__file__).parent / "fixtures" / "deepseek_v41_config.json").read_text())
EVIDENCE = json.loads((ROOT / "docs" / "evidence" / "deepseek-v41-name-coverage.json").read_text())


def test_all_reviewed_state_keys_have_graph_components_and_stable_evidence():
    expected = expected_state_keys(CONFIG)
    ir = ModelIR(MODEL_ID, "", Graph("empty", [], []), metadata={"artifacts": {}})
    assert (
        build_reviewed_topology(ModelReference(MODEL_ID, REVISION), REVISION, REQUIRED_FILES, CONFIG, ir)
        is ir
    )
    paths = {node.module_path for graph in ir.root_graph.walk() for node in graph.nodes}
    assert set(expected.values()) <= paths
    assert len(expected) == EVIDENCE["state_key_count"] == 96085
    assert dict(sorted(Counter(expected.values()).items())) == EVIDENCE["component_state_key_counts"]
    assert (
        dict(sorted(Counter(re.sub(r"(?<=\.)\d+(?=\.|$)", "*", name) for name in expected).items()))
        == EVIDENCE["normalized_patterns"]
    )
    for marker in ("image_start", "image_end", "image_newline"):
        node = next(node for node in ir.root_graph.nodes if node.module_path == marker)
        assert node.metadata["root_state_key"] == marker
        assert expected[marker] == marker
    assert expected["layers.14.ffn.experts.383.w3.scale"] == "layers.14.ffn.experts"
    assert expected["vision.blocks.31.attn.wqkv.bias"] == "vision.blocks[*].attn"
    assert "layers.14.attn.unreviewed.weight" not in expected
    assert "layers.40.attn.wq_a.weight" not in expected
    assert "mtp.3.ffn.gate.weight" not in expected


def test_converter_name_rules_are_explicit_and_do_not_assume_value_equivalence():
    assert converted_name("model.layers.0.self_attn.wq_a.weight_scale_inv") == ("layers.0.attn.wq_a.scale")
    assert converted_name("layers.0.mlp.gate.e_score_correction_bias") == ("layers.0.ffn.gate.bias")
    assert converted_name("vision.blocks.0.mlp.w1.weight") == "vision.blocks.0.mlp.w1.weight"
    assert converted_name("mtp.0.embed.weight") is None
    assert all(converted_name(name) == name for name in expected_state_keys(CONFIG))


@pytest.fixture
def audited_inputs(tmp_path, monkeypatch):
    from model_anatomy.src import publisher_name_coverage as names

    expected = expected_state_keys(CONFIG)
    index = {"weight_map": dict.fromkeys(expected, "model-00001-of-00001.safetensors")}
    monkeypatch.setattr(
        names, "INDEX_SHA256", hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sources = {}
    for filename in (
        "inference/config.json",
        "inference/model.py",
        "inference/vision.py",
        "inference/engram.py",
    ):
        body = filename.encode()
        (source_dir / Path(filename).name).write_bytes(body)
        sources[filename] = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(names, "REVIEWED_SOURCE_SHA256", sources)
    converter = tmp_path / "convert.py"
    converter.write_bytes(b"reviewed converter")
    monkeypatch.setattr(names, "CONVERTER_SHA256", hashlib.sha256(converter.read_bytes()).hexdigest())
    return index, source_dir, converter


def test_exact_name_audit_accepts_only_complete_identity(audited_inputs):
    index, source_dir, converter = audited_inputs
    result = audit_name_coverage(index, CONFIG, source_dir, converter)
    assert result["matched_state_key_count"] == 96085
    assert result["missing_state_key_count"] == result["unexpected_state_key_count"] == 0
    assert result["converter_renamed_names"] == 0
    assert result["converter_skipped_names"] == 0
    assert result["weight_payload_bytes_read"] == 0


@pytest.mark.parametrize("change", ["missing", "unexpected", "source", "converter", "index"])
def test_exact_name_audit_rejects_gaps_or_identity_change(audited_inputs, change):
    index, source_dir, converter = audited_inputs
    if change == "missing":
        index["weight_map"].pop("image_start")
    elif change == "unexpected":
        index["weight_map"]["layers.0.attn.unreviewed.weight"] = "model-00001-of-00001.safetensors"
    elif change == "source":
        (source_dir / "model.py").write_text("different")
    elif change == "converter":
        converter.write_text("different")
    else:
        index["metadata"] = {"different": True}
    with pytest.raises(ValueError, match="mismatch"):
        audit_name_coverage(index, CONFIG, source_dir, converter)


def test_failed_cli_audit_keeps_inspection_error_log(tmp_path):
    output = tmp_path / "names.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_publisher_checkpoint_names.py"),
        "--index",
        str(tmp_path / "missing-index.json"),
        "--config",
        str(Path(__file__).parent / "fixtures" / "deepseek_v41_config.json"),
        "--source-dir",
        str(tmp_path),
        "--converter",
        str(tmp_path / "missing-convert.py"),
        "--output",
        str(output),
    ]
    run = subprocess.run(command, check=False, capture_output=True, text=True)
    failure = json.loads(output.with_suffix(".failed.json").read_text())
    assert run.returncode == 1
    assert failure["status"] == "failed_unverified"
    assert failure["error_type"] == "FileNotFoundError"
    assert not output.exists()
