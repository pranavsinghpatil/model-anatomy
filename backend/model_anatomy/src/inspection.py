"""Compatibility interfaces backed by native Transformers static inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PretrainedConfig

from .adapters import adapter_for
from .artifacts import CheckpointMetadata, SafetensorsMetadata
from .errors import ArtifactError
from .facts import inspect_native_model
from .registry import instantiate_native, load_native_config


@dataclass(frozen=True)
class ModelDimensions:
    vocabulary: int | None
    context: int | None
    hidden: int | None
    layers: int | None
    heads: int | None
    inner: int | None
    activation: str | None

    @property
    def head_dimension(self) -> int | None:
        return self.hidden // self.heads if self.hidden and self.heads else None


GPT2Dimensions = ModelDimensions


def load_config(data: dict[str, Any]) -> tuple[PretrainedConfig, ModelDimensions]:
    config = load_native_config(data)
    for name in (
        "vocab_size",
        "n_positions",
        "n_embd",
        "n_layer",
        "n_head",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
    ):
        if name in data and (type(data[name]) is not int or data[name] <= 0):
            raise ArtifactError(f"Invalid configuration dimension: {name}")
    hidden = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
    heads = getattr(config, "num_attention_heads", None) or getattr(config, "num_heads", None)
    if (
        type(hidden) is int
        and type(heads) is int
        and hidden % heads
        and not getattr(config, "head_dim", None)
    ):
        raise ArtifactError("Hidden dimension is not divisible by attention heads")
    inner = (
        getattr(config, "intermediate_size", None)
        or getattr(config, "d_ff", None)
        or getattr(config, "n_inner", None)
    )
    if inner is None and type(hidden) is int:
        inner = 4 * hidden
    dimensions = ModelDimensions(
        getattr(config, "vocab_size", None),
        getattr(config, "max_position_embeddings", None),
        hidden,
        getattr(config, "num_hidden_layers", None) or getattr(config, "num_layers", None),
        heads,
        inner,
        getattr(config, "hidden_act", None) or getattr(config, "activation_function", None),
    )
    return config, dimensions


def instantiate(config: PretrainedConfig) -> torch.nn.Module:
    return instantiate_native(config)[0]


def inspect_model(
    model: torch.nn.Module, checkpoint: CheckpointMetadata | SafetensorsMetadata
) -> dict[str, Any]:
    raw = inspect_native_model(model, checkpoint)
    adapter = adapter_for(model.config.model_type)
    if adapter:
        adapter.refine_facts(raw, checkpoint, model.config.to_dict())
    return raw


def detect_repetition(raw: dict[str, Any], dimensions: ModelDimensions) -> list[str]:
    """Legacy single-stack validator; the general engine uses SemanticAnalyzer signatures."""
    modules, parameters = raw["modules"], raw["parameters"]
    container = next(
        (
            m
            for m in modules.values()
            if m["container"] == "module_list"
            and m["children"]
            and all("Block" in modules[p]["type"] or "Layer" in modules[p]["type"] for p in m["children"])
        ),
        None,
    )
    if container is None or len(container["children"]) != dimensions.layers:
        raise ArtifactError("Configured layer count disagrees with discovered blocks")
    paths = container["children"]

    def signature(prefix: str) -> tuple:
        return (
            tuple(
                (
                    p[len(prefix) :],
                    m["qualified_type"],
                    repr(m["attributes"]),
                    m.get("dropout_probability"),
                    m.get("epsilon"),
                )
                for p, m in modules.items()
                if p == prefix or p.startswith(prefix + ".")
            ),
            tuple(
                (p[len(prefix) :], tuple(v["shape"]), v["dtype"])
                for p, v in parameters.items()
                if p.startswith(prefix + ".")
            ),
        )

    if any(signature(path) != signature(paths[0]) for path in paths[1:]):
        raise ArtifactError("Blocks are not structurally equivalent")
    return paths
