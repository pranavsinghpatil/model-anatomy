"""Small optional refinements; adapters never construct complete model graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .artifacts import CheckpointMetadata, SafetensorsMetadata


@dataclass(frozen=True)
class ArchitectureAdapter:
    name: str
    semantic_aliases: dict[str, str] = field(default_factory=dict)
    node_aliases: dict[str, str] = field(default_factory=dict)
    graph_aliases: dict[str, str] = field(default_factory=dict)
    type_labels: dict[str, str] = field(default_factory=dict)
    input_aliases: dict[str, str] = field(default_factory=dict)

    def refine_facts(
        self,
        raw: dict[str, Any],
        checkpoint: CheckpointMetadata | SafetensorsMetadata,
        config: dict[str, Any],
    ) -> None:
        if self.name != "gpt2":
            return
        context = config.get("n_positions")
        for name, item in raw["checkpoint_extras"].items():
            module_path = name.removeprefix("transformer.").removesuffix(".bias")
            owner = "transformer." + module_path
            tensor = checkpoint.tensors[name]
            if (
                raw["modules"].get(owner, {}).get("type") == "GPT2Attention"
                and tensor.shape == (1, 1, context, context)
                and tensor.dtype in {"BOOL", "U8", "F32"}
            ):
                item.update(
                    kind="legacy_buffer",
                    evidence_kind="reconstructed",
                    module_path=owner,
                    reason="Historical GPT-2 causal-mask name and shape; values were not inspected",
                )


ADAPTERS = {
    "gpt2": ArchitectureAdapter(
        "gpt2",
        {"GPT2Block": "transformer_block", "GPT2Attention": "self_attention", "GPT2MLP": "feed_forward"},
        {"wte": "wte", "wpe": "wpe", "drop": "dropout", "h": "blocks", "ln_f": "ln_f", "attn": "attention"},
        {
            "GPT2Block": "transformer_block",
            "GPT2Attention": "attention",
            "GPT2MLP": "feed_forward",
            "attention_core": "attention_core",
        },
        {"GPT2Attention": "Multi-Head Attention"},
        {"input_ids": "tokens", "position_ids": "positions", "hidden_states": "input"},
    )
}


def adapter_for(model_type: str) -> ArchitectureAdapter | None:
    return ADAPTERS.get(model_type)
