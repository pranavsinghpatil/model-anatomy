"""Opt-in artifact inventory never fabricates a model graph."""

import io
import json

import pytest
import requests
import torch

from model_anatomy.src import limits, registry
from model_anatomy.src.artifacts import (
    AmbiguousCheckpointError,
    CheckpointMetadata,
    InspectionBudgetError,
    MetadataClient,
    parse_reference,
    parse_safetensors_header,
)
from model_anatomy.src.errors import ArtifactError, UnsupportedModelError
from model_anatomy.src.pipeline import reconstruct

SHA = "b" * 40
GPT2_CONFIG = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "vocab_size": 32,
    "n_positions": 16,
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 2,
}
UNKNOWN_CONFIG = {"model_type": "kolibri", "architectures": ["KolibriForCausalLM"]}


def checkpoint_with_tensor(name="layers.0.weight"):
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            name: {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        }
    ).encode()
    shard = parse_safetensors_header(header, 8 + len(header) + 24)
    return CheckpointMetadata(
        shard.tensors,
        {"model.safetensors": shard},
        {name: "model.safetensors"},
    )


class InspectionClient(MetadataClient):
    def __init__(self, files, config=None, checkpoint=None, adapter_config=None, checkpoint_error=None):
        super().__init__()
        self.files = set(files)
        self.config = config
        self.fixture_checkpoint = checkpoint
        self.adapter_config = adapter_config
        self.checkpoint_error = checkpoint_error

    def resolve(self, reference):
        self.repository_files[(reference.model_id, SHA)] = self.files
        return SHA

    def json(self, url, max_bytes=1024 * 1024):
        self.audit.append({"url": url, "bytes_read": 1, "status": 200})
        if url.endswith("/adapter_config.json"):
            return self.adapter_config or {"base_model_name_or_path": "base/model"}
        if self.config is None:
            raise requests.HTTPError("unexpected config read")
        return self.config

    def checkpoint(self, reference, revision):
        if self.checkpoint_error:
            raise self.checkpoint_error
        return self.fixture_checkpoint


def inspect(client, *, allow=True, value="org/model"):
    return reconstruct(value, client=client, allow_artifact_inspection=allow)


def assert_inventory_only(ir, reason):
    assert ir.architecture == "Unavailable"
    assert ir.root_graph.id == "architecture_unavailable"
    assert ir.root_graph.nodes == []
    assert ir.root_graph.edges == []
    assert ir.metadata["inspection"]["kind"] == "artifact_inspection"
    assert ir.metadata["inspection"]["reason_code"] == reason
    assert ir.metadata["analysis_report"] == {
        "mode": "artifact_inspection",
        "forward_executed": False,
        "runtime_trace_available": False,
    }
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    ir.root_graph.validate()


def test_unknown_native_architecture_has_only_verified_inventory():
    files = {"config.json", "model.safetensors", "README.md"}
    client = InspectionClient(files, UNKNOWN_CONFIG, checkpoint_with_tensor())
    with pytest.raises(UnsupportedModelError):
        inspect(client, allow=False)

    ir = inspect(InspectionClient(files, UNKNOWN_CONFIG, checkpoint_with_tensor()))
    assert_inventory_only(ir, "native_architecture_unavailable")
    assert ir.metadata["inspection"]["files"] == sorted(files)
    assert ir.metadata["artifacts"]["repository_files"] == sorted(files)
    assert ir.metadata["inspection"]["checkpoint_available"] is True
    assert ir.metadata["checkpoint_tensor_element_count"] == 6
    assert ir.metadata.get("parameter_count") is None
    assert ir.tensors["layers.0.weight"]["shape"] == [2, 3]
    assert ir.tensors["layers.0.weight"]["role"] == "unclassified"
    assert "semantic_coverage" not in ir.metadata["analysis_report"]


def test_foreign_diffusers_config_reaches_inventory_without_transformers_dimension_rules():
    client = InspectionClient(
        {"config.json", "diffusion_pytorch_model.safetensors"},
        {"_class_name": "UNet2DConditionModel", "num_attention_heads": None},
        checkpoint_with_tensor(),
    )

    ir = inspect(client)

    assert_inventory_only(ir, "native_architecture_unavailable")
    assert ir.metadata["model_profile"]["config_model_type"] is None


def test_forward_guard_remains_unsupported_and_pipeline_falls_back_to_inventory(monkeypatch):
    class ForwardProbe(torch.nn.Module):
        def forward(self, inputs):
            return inputs

    class FakeAutoModel:
        @classmethod
        def from_config(cls, config, trust_remote_code=False):
            model = ForwardProbe()
            model(torch.ones(1))
            return model

    monkeypatch.setattr(registry, "select_auto_class", lambda config: (FakeAutoModel, []))
    client = InspectionClient(
        {"config.json", "model.safetensors"},
        GPT2_CONFIG,
        checkpoint_with_tensor(),
    )

    ir = inspect(client)

    assert_inventory_only(ir, "native_architecture_unavailable")
    assert any(
        "Native construction attempted forward execution" in blocker
        for blocker in ir.metadata["model_profile"]["architecture_blockers"]
    )


def test_adapter_config_requires_base_and_keeps_header_evidence():
    files = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
    client = InspectionClient(files, checkpoint=checkpoint_with_tensor("base_model.lora_A.weight"))
    ir = inspect(client)
    assert_inventory_only(ir, "adapter_requires_base_model")
    assert ir.metadata["checkpoint_tensor_element_count"] == 6


def test_lora_tensor_names_are_not_assigned_to_a_native_graph():
    files = {"config.json", "model.safetensors"}
    ir = inspect(
        InspectionClient(
            files,
            GPT2_CONFIG,
            checkpoint_with_tensor("transformer.h.0.attn.lora_B.weight"),
        )
    )
    assert_inventory_only(ir, "adapter_requires_base_model")


def test_missing_config_can_still_inventory_an_unambiguous_checkpoint():
    files = {"model.safetensors", "README.md"}
    ir = inspect(InspectionClient(files, checkpoint=checkpoint_with_tensor()))
    assert_inventory_only(ir, "architecture_config_unavailable")
    assert "source_config" not in ir.metadata
    assert ir.tensors


def test_native_config_without_safetensors_still_reconstructs_the_module_graph():
    files = {"config.json", "pytorch_model.bin", "README.md"}
    ir = inspect(InspectionClient(files, GPT2_CONFIG))
    assert ir.architecture == "GPT2LMHeadModel"
    assert ir.metadata.get("inspection") is None
    assert ir.root_graph.nodes
    def has_embedding(graph):
        return any(
            node.kind == "embedding" or node.child_graph is not None and has_embedding(node.child_graph)
            for node in graph.nodes
        )

    assert has_embedding(ir.root_graph)
    assert ir.tensors == {}
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["warnings"] == [
        "No Safetensors checkpoint was available. The module graph comes from the native configuration; tensor ownership was not established."
    ]
    ir.root_graph.validate()


def test_explicit_missing_safetensors_file_is_not_downgraded_to_inventory():
    client = InspectionClient({"README.md"})
    value = f"https://huggingface.co/org/model/blob/{SHA}/missing.safetensors"

    with pytest.raises(ArtifactError, match="Requested Safetensors artifact is missing"):
        inspect(client, value=value)

    assert client.audit == []


def test_missing_config_and_safetensors_is_an_inventory_not_a_guess():
    files = {"README.md", "modeling_custom.py"}
    ir = inspect(InspectionClient(files))
    assert_inventory_only(ir, "architecture_config_unavailable")
    assert ir.metadata["inspection"]["files"] == sorted(files)
    assert ir.tensors == {}


@pytest.mark.parametrize(
    ("files", "config", "message"),
    [
        ({"README.md"}, None, "no config.json"),
        ({"config.json", "README.md"}, GPT2_CONFIG, "no Safetensors checkpoint"),
    ],
)
def test_strict_mode_still_rejects_incomplete_repository_layout(files, config, message):
    with pytest.raises(ArtifactError, match=message):
        inspect(InspectionClient(files, config), allow=False)


def test_multiple_checkpoint_components_are_not_arbitrarily_selected():
    files = {"config.json", "model.safetensors", "adapter_model.safetensors"}
    client = InspectionClient(
        files,
        GPT2_CONFIG,
        checkpoint_error=AmbiguousCheckpointError("ambiguous candidates"),
    )
    ir = inspect(client)
    assert_inventory_only(ir, "checkpoint_selection_required")
    assert ir.tensors == {}
    assert ir.metadata["checkpoint_tensor_element_count"] is None


def test_canonical_root_selection_survives_nested_training_artifacts():
    files = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
        "training/checkpoint.safetensors",
        "training/nested/model.safetensors.index.json",
        "README.md",
    }
    checkpoint = CheckpointMetadata(
        checkpoint_with_tensor("transformer.h.0.attn.lora_A.weight").tensors,
        {"model-00001-of-00001.safetensors": checkpoint_with_tensor().shards["model.safetensors"]},
        {"transformer.h.0.attn.lora_A.weight": "model-00001-of-00001.safetensors"},
        {"weight_map": {"transformer.h.0.attn.lora_A.weight": "model-00001-of-00001.safetensors"}},
    )
    ir = inspect(InspectionClient(files, GPT2_CONFIG, checkpoint))
    assert_inventory_only(ir, "adapter_requires_base_model")
    assert ir.metadata["inspection"]["files"] == sorted(files)
    assert ir.metadata["inspection"]["selected_checkpoint_files"] == [
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
    ]
    assert list(ir.tensors) == ["transformer.h.0.attn.lora_A.weight"]


@pytest.mark.parametrize(
    ("files", "index", "expected"),
    [
        (
            {"multiple_angles_step1000.safetensors", "training/step.safetensors"},
            None,
            {"multiple_angles_step1000.safetensors"},
        ),
        (
            {
                "model.safetensors.index.json",
                "model-00001-of-00001.safetensors",
                "training/step.safetensors.index.json",
                "training/step.safetensors",
            },
            {"weight_map": {"layers.0.weight": "model-00001-of-00001.safetensors"}},
            {"model.safetensors.index.json", "model-00001-of-00001.safetensors"},
        ),
    ],
)
def test_checkpoint_selection_prefers_canonical_or_sole_root(files, index, expected):
    client = MetadataClient()
    reference = parse_reference("org/model")
    client.repository_files[(reference.model_id, SHA)] = files
    client.json = lambda *args, **kwargs: index
    shard = checkpoint_with_tensor().shards["model.safetensors"]
    client.safetensors = lambda *args, **kwargs: shard

    checkpoint = client.checkpoint(reference, SHA)
    selected = set(checkpoint.tensor_files.values())
    if checkpoint.index is not None:
        selected.add("model.safetensors.index.json")
    assert selected == expected


def test_malformed_checkpoint_metadata_is_not_converted_to_fallback():
    client = InspectionClient(
        {"config.json", "model.safetensors"},
        UNKNOWN_CONFIG,
        checkpoint_error=ArtifactError("Invalid Safetensors JSON header"),
    )
    with pytest.raises(ArtifactError, match="Invalid Safetensors JSON header"):
        inspect(client)


def test_valid_checkpoint_budget_limit_becomes_inventory_only_when_opted_in():
    files = {"config.json", "model.safetensors.index.json", "one.safetensors", "README.md"}
    client = InspectionClient(
        files,
        UNKNOWN_CONFIG,
        checkpoint_error=InspectionBudgetError("Checkpoint index exceeds the supported tensor record count"),
    )
    with pytest.raises(InspectionBudgetError):
        inspect(
            InspectionClient(
                files,
                GPT2_CONFIG,
                checkpoint_error=InspectionBudgetError(
                    "Checkpoint index exceeds the supported tensor record count"
                ),
            ),
            allow=False,
        )
    ir = inspect(client)
    assert_inventory_only(ir, "metadata_budget_exceeded")
    assert ir.metadata["inspection"]["checkpoint_available"] is True
    assert ir.metadata["inspection"]["selected_checkpoint_files"] == []
    assert ir.metadata["checkpoint_tensor_element_count"] is None
    assert ir.metadata["source_config"] == UNKNOWN_CONFIG


def test_budgeted_index_retains_declared_counts_without_claiming_verified_tensors(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 2)
    files = {"config.json", "model.safetensors.index.json", "one.safetensors", "two.safetensors"}
    index = {"weight_map": {"a": "one.safetensors", "b": "one.safetensors", "c": "two.safetensors"}}

    class IndexedClient(InspectionClient):
        def json(self, url, max_bytes=1024 * 1024):
            self.audit.append({"url": url, "bytes_read": 100, "status": 200})
            return index if url.endswith(".index.json") else UNKNOWN_CONFIG

        checkpoint = MetadataClient.checkpoint

        def safetensors(self, url):
            pytest.fail("Shard headers must not be fetched after the index budget is exceeded")

    ir = inspect(IndexedClient(files, UNKNOWN_CONFIG))
    assert_inventory_only(ir, "metadata_budget_exceeded")
    declared = ir.metadata["artifacts"]["declared_checkpoint_index"]
    assert declared["declared_tensor_names"] == 3
    assert declared["declared_shards"] == 2
    assert declared["tensor_record_limit"] == 2
    assert declared["canonical_sha256"]
    assert ir.metadata["checkpoint_tensor_element_count"] is None
    assert ir.tensors == {}
    assert any(item.source == "hf_checkpoint_index" for item in ir.evidence)


def test_budgeted_index_does_not_hide_missing_shard(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 1)

    class InvalidIndexClient(InspectionClient):
        def json(self, url, max_bytes=1024 * 1024):
            return {"weight_map": {"a": "missing.safetensors", "b": "missing.safetensors"}} if url.endswith(".index.json") else UNKNOWN_CONFIG

        checkpoint = MetadataClient.checkpoint

    client = InvalidIndexClient({"config.json", "model.safetensors.index.json"}, UNKNOWN_CONFIG)
    with pytest.raises(ArtifactError, match="invalid or missing shard"):
        inspect(client)


def test_inventory_budget_audit_retains_bytes_read_before_the_limit():
    from model_anatomy.src.artifact_inspection import build_artifact_inspection

    ir = build_artifact_inspection(
        parse_reference("org/model"),
        SHA,
        {"model.safetensors"},
        [
            {
                "url": "https://huggingface.co/org/model/resolve/" + SHA + "/model.safetensors",
                "range": [0, 7],
                "bytes_read": 8,
                "status": 206,
            }
        ],
        audit_start=0,
        config_data=None,
        checkpoint=None,
        reason_code="metadata_budget_exceeded",
    )
    assert ir.metadata["artifacts"]["safetensors_bytes_read"] == 8
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.tensors == {}


def test_tensor_record_budget_does_not_mask_malformed_header(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 1)
    valid = {
        "first": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "second": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    encoded = json.dumps(valid).encode()
    with pytest.raises(InspectionBudgetError):
        parse_safetensors_header(encoded, 8 + len(encoded) + 8)

    malformed = {**valid, "second": {"dtype": "unknown", "shape": [1], "data_offsets": [4, 8]}}
    encoded = json.dumps(malformed).encode()
    with pytest.raises(ArtifactError, match="Unsupported Safetensors dtype"):
        parse_safetensors_header(encoded, 8 + len(encoded) + 8)


def test_resolver_retains_repositories_without_config_or_safetensors():
    class RawBody:
        def __init__(self, body):
            self.body = io.BytesIO(body)

        def read(self, size=-1, decode_content=False):
            return self.body.read(size)

    class Response:
        status_code = 200

        def __init__(self):
            self.raw = RawBody(
                json.dumps(
                    {"sha": SHA, "siblings": [{"rfilename": "README.md"}, {"rfilename": "modeling.py"}]}
                ).encode()
            )

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    client = MetadataClient(Session())
    reference = parse_reference("org/model")
    assert client.resolve(reference) == SHA
    assert client.repository_files[("org/model", SHA)] == {"README.md", "modeling.py"}
