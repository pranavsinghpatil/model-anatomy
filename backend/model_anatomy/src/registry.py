"""Use native Transformers Auto mappings instead of maintaining a model registry."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch
from transformers import AutoConfig, PretrainedConfig

from . import limits
from .errors import ArtifactError, UnsupportedModelError

_CONSTRUCTING = ContextVar("model_anatomy_meta_construction", default=False)


def _forbid_forward(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
    if _CONSTRUCTING.get():
        raise UnsupportedModelError("Native construction attempted forward execution; it is disabled")


_DIMENSION_FIELDS = {
    "vocab_size",
    "n_positions",
    "n_embd",
    "n_layer",
    "n_head",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
}
_LAYER_FIELDS = {
    "n_layer",
    "num_hidden_layers",
    "num_layers",
    "num_decoder_layers",
    "num_encoder_layers",
    "encoder_layers",
    "decoder_layers",
}
_STAGE_DEPTH_MODEL_TYPES = {"bit", "swin", "swinv2", "donut-swin", "maskformer-swin"}
_IMPLICIT_BACKBONE_STAGE_TYPES = {
    "dpt": "bit",
    "mask2former": "swin",
    "maskformer": "swin",
    "oneformer": "swin",
    "vit-hybrid": "bit",
}


def _validate_embedded_config(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ArtifactError("Native configuration must be an object")
    pending: list[tuple[Any, int, str | None]] = [(data, 0, None)]
    visited = 0
    while pending:
        value, depth, inherited_stage_type = pending.pop()
        if not isinstance(value, (dict, list)):
            continue
        visited += 1
        if visited > limits.MAX_CONFIG_CONTAINER_NODES:
            raise ArtifactError("Configuration exceeds the nested metadata node limit")
        if depth > limits.MAX_CONFIG_NESTING_DEPTH:
            raise ArtifactError("Configuration exceeds the nested metadata depth limit")
        if isinstance(value, list):
            pending.extend((item, depth + 1, None) for item in value if isinstance(item, (dict, list)))
            continue

        if value.get("auto_map"):
            raise UnsupportedModelError("Native configuration is required; remote code is disabled")
        if value.get("use_pretrained_backbone") or value.get("pretrained"):
            raise UnsupportedModelError(
                "Pretrained backbone initialization is disabled; weights must not be loaded"
            )
        if value.get("backbone") and value.get("backbone_config") is None:
            raise UnsupportedModelError(
                "External backbone resolution is disabled; embedded backbone_config is required"
            )
        model_type = value.get("model_type")
        stage_type = model_type if isinstance(model_type, str) else inherited_stage_type
        layer_fields = set(_LAYER_FIELDS)
        if isinstance(model_type, str):
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            if model_type in CONFIG_MAPPING:
                # Native configs expose aliases such as DistilBERT's n_layers.
                # Resolve those names before any AutoConfig/model construction.
                layer_fields.update(
                    alias
                    for canonical, alias in CONFIG_MAPPING[model_type].attribute_map.items()
                    if canonical in _LAYER_FIELDS
                )
        if stage_type in _STAGE_DEPTH_MODEL_TYPES and "depths" in value:
            depths = value["depths"]
            if not isinstance(depths, list) or not depths:
                raise ArtifactError("Invalid configuration layer schedule: depths")
            if len(depths) > limits.MAX_MODEL_LAYERS:
                raise UnsupportedModelError(
                    f"Model exceeds the supported layer count ({limits.MAX_MODEL_LAYERS})"
                )
            if any(type(depth) is not int or depth <= 0 for depth in depths):
                raise ArtifactError("Invalid configuration layer schedule: depths")
            if sum(depths) > limits.MAX_MODEL_LAYERS:
                raise UnsupportedModelError(
                    f"Model exceeds the supported layer count ({limits.MAX_MODEL_LAYERS})"
                )
        for name in _DIMENSION_FIELDS | layer_fields:
            if name not in value:
                continue
            dimension = value[name]
            if type(dimension) is not int or dimension <= 0:
                raise ArtifactError(f"Invalid configuration dimension: {name}")
            if name in layer_fields and dimension > limits.MAX_MODEL_LAYERS:
                raise UnsupportedModelError(
                    f"Model exceeds the supported layer count ({limits.MAX_MODEL_LAYERS})"
                )
        implicit_stage_type = None
        if isinstance(model_type, str):
            implicit_stage_type = _IMPLICIT_BACKBONE_STAGE_TYPES.get(model_type)
            if model_type == "dpt" and not value.get("is_hybrid"):
                implicit_stage_type = None
        pending.extend(
            (
                item,
                depth + 1,
                implicit_stage_type if name == "backbone_config" else None,
            )
            for name, item in value.items()
            if isinstance(item, (dict, list))
        )


def load_native_config(data: dict[str, Any]) -> PretrainedConfig:
    _validate_embedded_config(data)
    model_type = data.get("model_type")
    if not isinstance(model_type, str) or data.get("auto_map"):
        raise UnsupportedModelError("A native Transformers config is required; remote code is disabled")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type not in CONFIG_MAPPING:
        raise UnsupportedModelError(f"No native Transformers config for model_type={model_type}")
    try:
        config = AutoConfig.for_model(model_type, **{k: v for k, v in data.items() if k != "model_type"})
    except (KeyError, ValueError, TypeError) as exc:
        raise ArtifactError(f"Invalid native configuration: {exc}") from exc
    except RecursionError as exc:
        raise ArtifactError("Native configuration is too deeply nested") from exc
    if not type(config).__module__.startswith("transformers."):
        raise UnsupportedModelError("Third-party config implementations are outside the target scope")
    return config


def select_auto_class(config: PretrainedConfig) -> tuple[type, list[str]]:
    architectures = config.architectures
    if architectures is None:
        architectures = []
    if not isinstance(architectures, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in architectures
    ):
        raise ArtifactError("Invalid native configuration architectures; expected a list of class names")

    candidates = _auto_model_candidates(config)
    matches = [candidate for candidate in candidates if candidate[2] in architectures]
    if matches:
        preferred = _task_routing_hint(config)
        matches.sort(
            key=lambda candidate: (
                candidate[0] != preferred,
                candidate[0] != "AutoModel",
                candidate[0],
            )
        )
        return matches[0][1], []

    if architectures:
        declared = ", ".join(architectures)
        raise UnsupportedModelError(f"No native AutoModel mapping for declared architecture(s): {declared}")

    hint = _task_routing_hint(config)
    if hint is not None:
        routed = next((candidate for candidate in candidates if candidate[0] == hint), None)
        if routed is not None:
            return routed[1], [f"No architecture declared; selected {hint} from native config task metadata"]

    backbone = next((candidate for candidate in candidates if candidate[0] == "AutoModel"), None)
    if backbone is None:
        raise UnsupportedModelError(f"No native PyTorch AutoModel for {type(config).__name__}")
    return backbone[1], [
        "No architecture declared or compatible task wrapper found; selected the native AutoModel backbone"
    ]


def _auto_model_candidates(config: PretrainedConfig) -> list[tuple[str, type, str]]:
    """Read upstream Auto mappings as names without resolving model implementations."""
    from transformers.models.auto import modeling_auto

    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str):
        return []

    candidates = []
    for name, auto_class in vars(modeling_auto).items():
        if not name.startswith("AutoModel") or not isinstance(auto_class, type):
            continue
        mapping = auto_class.__dict__.get("_model_mapping")
        mapping_names = getattr(mapping, "_model_mapping", None)
        if not isinstance(mapping_names, dict):
            continue
        model_names = mapping_names.get(model_type)
        if isinstance(model_names, str):
            candidates.append((name, auto_class, model_names))
        elif isinstance(model_names, (list, tuple)) and all(
            isinstance(model_name, str) for model_name in model_names
        ):
            candidates.extend((name, auto_class, model_name) for model_name in model_names)
    return candidates


def _task_routing_hint(config: PretrainedConfig) -> str | None:
    """Use generic native config modality flags only when architecture names do not decide."""
    if not getattr(config, "is_encoder_decoder", False):
        return None
    if getattr(config, "num_mel_bins", None) is not None:
        return "AutoModelForSpeechSeq2Seq"
    return "AutoModelForSeq2SeqLM"


def instantiate_native(config: PretrainedConfig) -> tuple[torch.nn.Module, dict[str, Any]]:
    auto_class, warnings = select_auto_class(config)
    # Eager implementation exposes inspectable source; no attention is executed.
    config._attn_implementation = "eager"
    token = _CONSTRUCTING.set(True)
    handle = torch.nn.modules.module.register_module_forward_pre_hook(_forbid_forward)
    try:
        with torch.device("meta"):
            model = auto_class.from_config(config, trust_remote_code=False)
            model.tie_weights()
            model.eval()
    except (ValueError, TypeError) as exc:
        raise ArtifactError(f"Invalid native model configuration for {config.model_type}: {exc}") from exc
    except (ImportError, RuntimeError, NotImplementedError) as exc:
        raise UnsupportedModelError(
            f"Native meta construction failed for {config.model_type}: {exc}"
        ) from exc
    finally:
        handle.remove()
        _CONSTRUCTING.reset(token)
    if not type(model).__module__.startswith("transformers."):
        raise UnsupportedModelError("Only native Transformers PyTorch models are supported")
    if any(t.device.type != "meta" for t in (*model.parameters(), *model.buffers())):
        raise ArtifactError(
            "Native construction allocated non-meta model tensors; no CPU fallback is allowed"
        )
    return model, {
        "auto_class": auto_class.__name__,
        "model_class": type(model).__name__,
        "trust_remote_code": False,
        "device": "meta",
        "warnings": warnings,
    }
