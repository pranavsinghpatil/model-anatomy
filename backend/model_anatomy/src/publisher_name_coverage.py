"""Exact state-key inventory for the reviewed DeepSeek publisher reference.

This describes source-declared checkpoint names, not tensor shapes, byte counts,
runtime dataflow, or the effects of model-parallel conversion on tensor values.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .checkpoint_audit_binding import INDEX_SHA256
from .publisher_topology import CONFIG_CANONICAL_SHA256, REVIEWED_SOURCE_SHA256

CONVERTER_SHA256 = "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"
SHARD_NAME = re.compile(r"model-\d{5}-of-\d{5}\.safetensors\Z")


def expected_state_keys(config: dict[str, Any]) -> dict[str, str]:
    """Map each reviewed publisher state key to its closest graph component.

    The finite inventory follows the constructors in the pinned model.py and
    vision.py. It intentionally rejects an extra key even under a known prefix.
    """
    text = config["text_config"]
    vision = config["vision_config"]
    result: dict[str, str] = {}

    def add(name: str, component: str) -> None:
        if name in result:
            raise ValueError(f"Duplicate expected state key: {name}")
        result[name] = component

    for name, component in (
        ("embed.weight", "embed"),
        ("norm.weight", "norm"),
        ("head.weight", "head"),
        ("image_start", "image_start"),
        ("image_end", "image_end"),
        ("image_newline", "image_newline"),
    ):
        add(name, component)
    for layer in ("w1", "w2"):
        for part in ("weight", "bias"):
            add(f"aligner.{layer}.{part}", "aligner")

    for part in ("weight", "bias"):
        add(f"vision.patch_embed.proj.{part}", "vision.patch_embed")
    add("vision.norm.weight", "vision.norm")
    for index in range(vision["num_hidden_layers"]):
        prefix = f"vision.blocks.{index}"
        component = "vision.blocks[*]"
        for norm in ("norm1", "norm2"):
            add(f"{prefix}.{norm}.weight", f"{component}.{norm}")
        for projection in ("wqkv", "wo"):
            for part in ("weight", "bias"):
                add(f"{prefix}.attn.{projection}.{part}", f"{component}.attn")
        for projection in ("w1", "w2"):
            add(f"{prefix}.mlp.{projection}.weight", f"{component}.mlp")

    for branch, blocks, experts in (
        ("layers", text["num_hidden_layers"], text["n_routed_experts"]),
        ("mtp", text["num_nextn_predict_layers"], text["dspark_n_routed_experts"]),
    ):
        for index in range(blocks):
            block = f"{branch}.{index}"
            for norm in ("attn_norm", "ffn_norm"):
                add(f"{block}.{norm}.weight", f"{block}.{norm}")
            for part in (
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            ):
                add(f"{block}.{part}", block)
            attention = f"{block}.attn"
            add(f"{attention}.attn_sink", attention)
            for norm in ("q_norm", "kv_norm"):
                add(f"{attention}.{norm}.weight", f"{attention}.{norm}")
            for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                for part in ("weight", "scale"):
                    add(f"{attention}.{projection}.{part}", f"{attention}.{projection}")

            if branch == "layers":
                if index in text["kv_source_layer_ids"]:
                    compressor = f"{attention}.compressor"
                    for suffix in ("norm.weight", "wkv.weight"):
                        add(f"{compressor}.{suffix}", compressor)
                    if text["compress_ratios"][index] > 1:
                        add(f"{compressor}.wgate.weight", compressor)
                if index in text["index_source_layer_ids"]:
                    indexer = f"{attention}.indexer"
                    for suffix in ("weights_proj.weight", "wq_b.weight", "wq_b.scale"):
                        add(f"{indexer}.{suffix}", indexer)
                    if index in text["kv_source_layer_ids"]:
                        for suffix in ("wk.weight", "k_norm.weight"):
                            add(f"{indexer}.{suffix}", indexer)
                if index in text["engram_layer_ids"]:
                    engram = f"{block}.engram"
                    for suffix in (
                        "embed.weight",
                        "embed.scale",
                        "q_weight",
                        "k_weight",
                        "wkv.weight",
                        "wkv.scale",
                    ):
                        add(f"{engram}.{suffix}", engram)

            ffn = f"{block}.ffn"
            for part in ("weight", "bias", "bias_vl"):
                add(f"{ffn}.gate.{part}", f"{ffn}.gate")
            for projection in ("w1", "w2", "w3"):
                for part in ("weight", "scale"):
                    add(f"{ffn}.shared_experts.{projection}.{part}", f"{ffn}.shared_experts")
            for expert in range(experts):
                for projection in ("w1", "w2", "w3"):
                    for part in ("weight", "scale"):
                        add(f"{ffn}.experts.{expert}.{projection}.{part}", f"{ffn}.experts")

            if branch == "mtp" and index == 0:
                for part in ("weight", "scale"):
                    add(f"{block}.main_proj.{part}", f"{block}.main_proj")
                add(f"{block}.main_norm.weight", block)
            if branch == "mtp" and index == blocks - 1:
                add(f"{block}.norm.weight", block)
                for suffix in ("embed.weight", "head.weight"):
                    add(f"{block}.markov_head.{suffix}", f"{block}.markov_head")
                add(f"{block}.confidence_head.proj.weight", f"{block}.confidence_head")
    return result


def converted_name(name: str) -> str | None:
    """Reproduce only the pinned convert.py state-key rewrites, without execution."""
    result = name.removeprefix("model.")
    if result.startswith("mtp.") and result.split(".", 2)[-1] in ("embed.weight", "head.weight"):
        return None
    result = result.replace("self_attn", "attn")
    if not result.startswith("vision."):
        result = result.replace("mlp", "ffn")
    return result.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")


def audit_name_coverage(
    index: dict[str, Any], config: dict[str, Any], source_dir: Path, converter: Path
) -> dict[str, Any]:
    """Require exact index, config, source hashes and a complete state-key set."""
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    index_digest = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    if config_digest != CONFIG_CANONICAL_SHA256 or index_digest != INDEX_SHA256:
        raise ValueError("Pinned config or checkpoint index digest mismatch")
    for name, digest in REVIEWED_SOURCE_SHA256.items():
        source = source_dir / Path(name).name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned publisher source digest mismatch: {name}")
    if hashlib.sha256(converter.read_bytes()).hexdigest() != CONVERTER_SHA256:
        raise ValueError("Pinned publisher converter digest mismatch")
    names = index.get("weight_map")
    if (
        not isinstance(names, dict)
        or not names
        or any(
            not isinstance(name, str) or not isinstance(shard, str) or not SHARD_NAME.fullmatch(shard)
            for name, shard in names.items()
        )
    ):
        raise ValueError("Invalid pinned checkpoint weight map")
    expected = expected_state_keys(config)
    missing = sorted(expected.keys() - names.keys())
    unexpected = sorted(names.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            f"State-key mismatch: {len(missing)} missing {missing[:5]}, "
            f"{len(unexpected)} unexpected {unexpected[:5]}"
        )
    converted = [converted_name(name) for name in names]
    if any(name is None for name in converted) or len(set(converted)) != len(converted):
        raise ValueError("Publisher conversion skips or collides on pinned state keys")
    components = dict(sorted(Counter(expected.values()).items()))
    patterns = dict(sorted(Counter(re.sub(r"(?<=\.)\d+(?=\.|$)", "*", name) for name in names).items()))
    return {
        "basis": "reviewed_publisher_state_key_inventory",
        "index_sha256": index_digest,
        "config_canonical_sha256": config_digest,
        "reviewed_sources_sha256": REVIEWED_SOURCE_SHA256,
        "converter_sha256": CONVERTER_SHA256,
        "state_key_count": len(names),
        "matched_state_key_count": len(expected),
        "missing_state_key_count": 0,
        "unexpected_state_key_count": 0,
        "index_shard_count": len(set(names.values())),
        "component_count": len(components),
        "normalized_pattern_count": len(patterns),
        "converter_renamed_names": sum(
            before != after for before, after in zip(names, converted, strict=True)
        ),
        "converter_skipped_names": 0,
        "converter_name_collisions": 0,
        "weight_payload_bytes_read": 0,
        "component_state_key_counts": components,
        "normalized_patterns": patterns,
        "limitations": [
            "State-key correspondence does not establish tensor shapes, element totals, parameter count, or runtime dataflow.",
            "The publisher converter can partition, pad, dequantize, and recast tensor values; its output has not been executed or inspected.",
        ],
    }
