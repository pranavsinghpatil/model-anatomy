"""Resolve, inspect, reconcile, and reconstruct one real GPT-2 checkpoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import requests

from .artifacts import MetadataClient, parse_reference, require_target
from .inspection import inspect_model, instantiate, load_config
from .ir import Evidence, ModelIR
from .semantic import build_semantic_ir


def reconstruct(value: str, *, client: MetadataClient | None = None) -> ModelIR:
    """Build a complete IR without downloading or materializing weight values."""
    reference = parse_reference(value)
    require_target(reference)
    if client is None:
        with requests.Session() as session:
            return reconstruct(value, client=MetadataClient(session))
    audit_start = len(client.audit)
    revision = client.resolve(reference)
    config_url = reference.url("config.json", revision)
    weights_url = reference.url(reference.filename, revision)
    config_data = client.json(config_url)
    config, dimensions = load_config(config_data)
    checkpoint = client.safetensors(weights_url)
    model = instantiate(config)
    raw = inspect_model(model, checkpoint)
    tensors = {}
    tensor_parameters: dict[str, list[str]] = {}
    for name, parameter in raw["parameters"].items():
        tensor_parameters.setdefault(parameter["tensor_name"], []).append(name)
    for name, tensor in sorted(checkpoint.tensors.items()):
        tensors[name] = {
            **tensor.public_dict(),
            "artifact": weights_url,
            "parameter_refs": tensor_parameters.get(name, []),
            "role": "parameter" if name in tensor_parameters else raw["checkpoint_extras"][name]["kind"],
            "evidence_kind": "direct",
            "role_evidence_kind": "reconstructed",
        }
    metadata = {
        "revision": revision,
        "requested_revision": reference.revision,
        "dimensions": asdict(dimensions),
        "source_config": config_data,
        "unique_parameter_count": raw["unique_parameter_count"],
        "checkpoint_tensor_element_count": sum(t.parameter_count for t in checkpoint.tensors.values()),
        "checkpoint_buffer_element_count": sum(
            t.parameter_count for n, t in checkpoint.tensors.items() if n in raw["checkpoint_extras"]
        ),
        "versions": raw["versions"],
        "parameter_count_policy": "Unique model parameters use object identity; component counts overlap across hierarchy and tied aliases",
        "artifacts": {
            "config_url": config_url,
            "safetensors_url": weights_url,
            "config_canonical_sha256": hashlib.sha256(
                json.dumps(config_data, sort_keys=True).encode()
            ).hexdigest(),
            "header_sha256": checkpoint.header_sha256,
            "header_length": checkpoint.header_length,
            "file_size": checkpoint.file_size,
            "safetensors_bytes_read": sum(r["bytes_read"] for r in checkpoint.requests),
            "weight_payload_bytes_read": 0,
            "requests": client.audit[audit_start:],
        },
        "limitations": [
            "Only openai-community/gpt2 and its built-in GPT2LMHeadModel are supported",
            "No weights, runtime execution, FX tracing, or numerical behavior validation",
            "Dataflow and symbolic shapes are GPT-2-specific reconstructions, not runtime observations",
            "Graphs describe eval-mode full-sequence inference without KV cache, padding masks, token types, or cross-attention",
            "Collapsed child graphs refer to block 0; instance-specific parameter records remain in raw_structure",
            "Legacy checkpoint mask buffers are identified by name and shape, without inspecting values",
        ],
    }
    evidence = [
        Evidence("hf_config", "direct", config_url, {"revision": revision}),
        Evidence(
            "safetensors_header",
            "direct",
            weights_url,
            {"header_sha256": checkpoint.header_sha256, "payload_bytes_read": 0},
        ),
        Evidence(
            "meta_model",
            "direct",
            "AutoModelForCausalLM.from_config",
            {"device": "meta", "remote_code": False},
        ),
        Evidence("parameter_identity", "direct", "lm_head.weight", {"shared_with": "transformer.wte.weight"}),
    ]
    ir = build_semantic_ir(reference.model_id, dimensions, raw, metadata, tensors, evidence)
    ir.root_graph.validate()
    return ir
