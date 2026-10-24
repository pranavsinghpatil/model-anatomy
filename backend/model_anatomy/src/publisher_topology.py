"""Reviewed publisher-reference topology for an exact, immutable model revision.

This route describes source-declared modules. It does not execute publisher code,
read weight payloads, or equate reference module paths with checkpoint ownership.
"""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise

from .artifacts import ModelReference
from .ir import Edge, Evidence, Graph, ModelIR, Node, RepetitionSpec
from .version import ENGINE_VERSION

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
CONFIG_CANONICAL_SHA256 = "0110b6f9349cf6404bef91bfe8a491be248d3d5a044592d3c443dc9073fa0096"
REVIEWED_SOURCE_SHA256 = {
    "inference/config.json": "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809",
    "inference/model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
    "inference/vision.py": "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "inference/engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
}
REQUIRED_FILES = set(REVIEWED_SOURCE_SHA256) | {"config.json", "model.safetensors.index.json"}


def _source(reference: ModelReference, filename: str) -> Evidence:
    return Evidence(
        "publisher_reference_source",
        "derived",
        reference.url(filename, REVISION),
        {
            "reviewed_sha256": REVIEWED_SOURCE_SHA256[filename],
            "scope": "reviewed_source_topology_only; source_not_executed",
        },
    )


def _node(
    node_id: str,
    label: str,
    kind: str,
    path: str,
    evidence: Evidence,
    *,
    child: Graph | None = None,
    repeat: int = 1,
    details: dict | None = None,
) -> Node:
    return Node(
        id=node_id,
        label=label,
        kind=kind,
        module_path=path,
        module_type="Publisher reference module",
        repeat=repeat,
        repetition=RepetitionSpec(repeat, "parallel") if repeat > 1 else None,
        metadata={
            "provenance": "publisher_reference_source",
            "parameter_count_status": "unverified",
            **(details or {}),
        },
        evidence=[evidence],
        child_graph=child,
    )


def _order(nodes: list[Node], evidence: Evidence) -> list[Edge]:
    return [
        Edge(
            left.id, right.id, None, label="reference order", relation="reference_order", evidence=[evidence]
        )
        for left, right in pairwise(nodes)
    ]


def _moe(reference: ModelReference, path: str, expert_count: int, active_count: int) -> Graph:
    source = _source(reference, "inference/model.py")
    nodes = [
        _node(
            "gate",
            "Router gate",
            "router",
            f"{path}.gate",
            source,
            details={"selected_experts_per_token": active_count},
        ),
        _node(
            "experts",
            "Routed experts",
            "expert",
            f"{path}.experts",
            source,
            repeat=expert_count,
            details={"declared_expert_count": expert_count, "distributed_rank_ownership": "not_inspected"},
        ),
        _node("shared", "Shared expert", "expert", f"{path}.shared_experts", source),
    ]
    return Graph(
        f"reference:{path}", nodes, [], {"provenance": "publisher_reference_source", "flow_verified": False}
    )


def _block(reference: ModelReference, path: str, layer: int, text: dict, *, mtp: bool = False) -> Graph:
    source = _source(reference, "inference/model.py")
    expert_count = text["dspark_n_routed_experts"] if mtp else text["n_routed_experts"]
    active_count = text["dspark_num_experts_per_tok"] if mtp else text["num_experts_per_tok"]
    nodes = []
    if not mtp and layer in text["engram_layer_ids"]:
        nodes.append(
            _node(
                "engram",
                "Engram memory",
                "memory",
                f"{path}.engram",
                source,
                details={"conditional_on_text_tokens": True},
            )
        )
    nodes.extend(
        [
            _node("attn_norm", "Attention norm", "normalization", f"{path}.attn_norm", source),
            _node(
                "attn",
                "Attention",
                "attention",
                f"{path}.attn",
                source,
                child=_attention(reference, f"{path}.attn", layer, text, mtp=mtp),
                details={"compress_ratio": text["compress_ratios"][layer]},
            ),
            _node("ffn_norm", "Feed-forward norm", "normalization", f"{path}.ffn_norm", source),
            _node(
                "ffn",
                "Mixture of experts",
                "mixture_of_experts",
                f"{path}.ffn",
                source,
                child=_moe(reference, f"{path}.ffn", expert_count, active_count),
            ),
        ]
    )
    if mtp and layer == text["num_hidden_layers"]:
        nodes.append(_node("main_proj", "Main hidden projection", "projection", f"{path}.main_proj", source))
    if mtp and layer == text["num_hidden_layers"] + text["num_nextn_predict_layers"] - 1:
        nodes.extend(
            [
                _node("markov_head", "Markov head", "head", f"{path}.markov_head", source),
                _node("confidence_head", "Confidence head", "head", f"{path}.confidence_head", source),
            ]
        )
    core_count = 5 if not mtp and layer in text["engram_layer_ids"] else 4
    return Graph(
        f"reference:{path}",
        nodes,
        _order(nodes[:core_count], source),
        {
            "provenance": "publisher_reference_source",
            "flow_verified": False,
            "hyper_connections": text["hc_mult"],
        },
    )


def _attention(reference: ModelReference, path: str, layer: int, text: dict, *, mtp: bool) -> Graph:
    source = _source(reference, "inference/model.py")
    nodes = [
        _node(
            name, label, "projection" if name.startswith("w") else "normalization", f"{path}.{name}", source
        )
        for name, label in (
            ("wq_a", "Query low-rank A"),
            ("q_norm", "Query norm"),
            ("wq_b", "Query low-rank B"),
            ("wkv", "Key/value projection"),
            ("kv_norm", "Key/value norm"),
            ("wo_a", "Output low-rank A"),
            ("wo_b", "Output low-rank B"),
        )
    ]
    if not mtp and layer in text["kv_source_layer_ids"]:
        nodes.append(_node("compressor", "KV compressor", "compression", f"{path}.compressor", source))
    if not mtp and layer in text["index_source_layer_ids"]:
        nodes.append(_node("indexer", "Sparse indexer", "indexer", f"{path}.indexer", source))
    return Graph(
        f"reference:{path}", nodes, [], {"provenance": "publisher_reference_source", "flow_verified": False}
    )


def _vision(reference: ModelReference, vision: dict) -> Graph:
    source = _source(reference, "inference/vision.py")
    block = Graph(
        "reference:vision.blocks[*]",
        [
            _node("norm1", "Attention norm", "normalization", "vision.blocks[*].norm1", source),
            _node("attn", "Vision attention", "attention", "vision.blocks[*].attn", source),
            _node("norm2", "MLP norm", "normalization", "vision.blocks[*].norm2", source),
            _node("mlp", "Vision MLP", "mlp", "vision.blocks[*].mlp", source),
        ],
        [],
        {"provenance": "publisher_reference_source", "flow_verified": False},
    )
    nodes = [
        _node("patch_embed", "Patch embedding", "embedding", "vision.patch_embed", source),
        _node(
            "blocks",
            "Vision blocks",
            "encoder_block",
            "vision.blocks",
            source,
            child=block,
            repeat=vision["num_hidden_layers"],
            details={"declared_layer_count": vision["num_hidden_layers"]},
        ),
        _node("norm", "Vision output norm", "normalization", "vision.norm", source),
    ]
    return Graph(
        "reference:vision",
        nodes,
        _order(nodes, source),
        {"provenance": "publisher_reference_source", "flow_verified": False},
    )


def build_reviewed_topology(
    reference: ModelReference,
    revision: str,
    files: set[str],
    config: dict,
    artifact_ir: ModelIR,
) -> ModelIR | None:
    """Return a partial graph only for the exact source-reviewed config and pin."""
    if reference.model_id != MODEL_ID or revision != REVISION or not REQUIRED_FILES <= files:
        return None
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if digest != CONFIG_CANONICAL_SHA256:
        return None
    text, vision = config["text_config"], config["vision_config"]
    source = _source(reference, "inference/model.py")
    nodes = [
        _node(
            "engram_hash",
            "N-gram hash state",
            "memory",
            "engram_hash",
            _source(reference, "inference/engram.py"),
        ),
        _node("embed", "Token embedding", "embedding", "embed", source),
    ]
    layers = []
    for layer in range(text["num_hidden_layers"]):
        path = f"layers.{layer}"
        layers.append(
            _node(
                f"layer_{layer}",
                f"Decoder block {layer}",
                "decoder_block",
                path,
                source,
                child=_block(reference, path, layer, text),
                details={"layer_index": layer},
            )
        )
    nodes.extend(layers)
    nodes.extend(
        [
            _node("norm", "Final norm", "normalization", "norm", source),
            _node("head", "Language head", "head", "head", source),
        ]
    )
    for layer in range(text["num_nextn_predict_layers"]):
        path = f"mtp.{layer}"
        nodes.append(
            _node(
                f"mtp_{layer}",
                f"Speculative block {layer}",
                "decoder_block",
                path,
                source,
                child=_block(reference, path, text["num_hidden_layers"] + layer, text, mtp=True),
                details={"separate_forward_spec_path": True},
            )
        )
    vision_source = _source(reference, "inference/vision.py")
    nodes.extend(
        [
            _node(
                "vision",
                "Vision transformer",
                "vision_encoder",
                "vision",
                vision_source,
                child=_vision(reference, vision),
                details={"conditional_on_images": True},
            ),
            _node(
                "aligner",
                "Image aligner",
                "projection",
                "aligner",
                vision_source,
                details={"conditional_on_images": True},
            ),
        ]
    )
    for marker in ("image_start", "image_end", "image_newline"):
        nodes.append(
            _node(
                marker,
                marker.replace("_", " ").title(),
                "parameter",
                marker,
                source,
                details={"conditional_on_images": True, "root_state_key": marker},
            )
        )
    root_edges = _order(nodes[1:2] + layers + nodes[2 + len(layers) : 4 + len(layers)], source)
    mtp_nodes = [node for node in nodes if node.id.startswith("mtp_")]
    root_edges += _order(mtp_nodes, source)
    root_edges += [
        Edge(
            "vision",
            "aligner",
            None,
            label="reference order",
            relation="reference_order",
            evidence=[vision_source],
        )
    ]
    graph = Graph(
        "publisher_reference_topology",
        nodes,
        root_edges,
        {
            "provenance": "publisher_reference_source",
            "flow_verified": False,
            "checkpoint_ownership_verified": False,
        },
    )
    graph.validate()
    artifact_ir.architecture = config["architectures"][0]
    artifact_ir.root_graph = graph
    artifact_ir.metadata.pop("inspection", None)
    artifact_ir.metadata["source_topology"] = {
        "basis": "exact_revision_reviewed_publisher_reference",
        "config_canonical_sha256": digest,
        "reviewed_sources": REVIEWED_SOURCE_SHA256,
        "checkpoint_headers_verified": False,
    }
    artifact_ir.metadata["dimensions"] = {
        "layers": text["num_hidden_layers"],
        "hidden": text["hidden_size"],
        "heads": text["num_attention_heads"],
        "vocabulary": text["vocab_size"],
        "vision_layers": vision["num_hidden_layers"],
    }
    artifact_ir.metadata["analysis_report"] = {
        "mode": "publisher_reference_topology",
        "forward_executed": False,
        "runtime_trace_available": False,
        "warnings": [
            "This partial graph is from reviewed publisher reference source, not a constructed native model.",
            "Checkpoint shard headers and tensor ownership have not been verified; parameter counts and dataflow are unavailable.",
        ],
    }
    artifact_ir.metadata["limitations"] = [
        "The source topology is pinned to one reviewed revision and exact configuration.",
        "Reference-order links are not runtime dataflow or shape evidence.",
        "The declared index exceeds the tensor-record budget; checkpoint shard headers remain unverified.",
    ]
    artifact_ir.metadata["engine_version"] = ENGINE_VERSION
    artifact_ir.evidence.extend(_source(reference, name) for name in REVIEWED_SOURCE_SHA256)
    from .checkpoint_audit_binding import bind_pinned_header_audit

    bind_pinned_header_audit(artifact_ir, reference, revision, files)
    return artifact_ir
