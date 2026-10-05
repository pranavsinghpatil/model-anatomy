"""Use native Transformers Auto mappings instead of maintaining a model registry."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModel, PretrainedConfig

from .errors import ArtifactError, UnsupportedModelError

_CONSTRUCTING = ContextVar("model_anatomy_meta_construction", default=False)


def _forbid_forward(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
    if _CONSTRUCTING.get():
        raise UnsupportedModelError("Native construction attempted forward execution; it is disabled")


def _validate_embedded_config(data: dict[str, Any]) -> None:
    if data.get("auto_map"):
        raise UnsupportedModelError("Native configuration is required; remote code is disabled")
    if data.get("use_pretrained_backbone") or data.get("pretrained"):
        raise UnsupportedModelError(
            "Pretrained backbone initialization is disabled; weights must not be loaded"
        )
    if data.get("backbone") and data.get("backbone_config") is None:
        raise UnsupportedModelError(
            "External backbone resolution is disabled; embedded backbone_config is required"
        )
    for value in data.values():
        if isinstance(value, dict):
            _validate_embedded_config(value)


def load_native_config(data: dict[str, Any]) -> PretrainedConfig:
    _validate_embedded_config(data)
    model_type = data.get("model_type")
    if not isinstance(model_type, str) or data.get("auto_map"):
        raise UnsupportedModelError("A native Transformers config is required; remote code is disabled")
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
    try:
        config = AutoConfig.for_model(model_type, **{k: v for k, v in data.items() if k != "model_type"})
    except (KeyError, ValueError) as exc:
        raise UnsupportedModelError(f"No native Transformers config for model_type={model_type}") from exc
    except TypeError as exc:
        raise ArtifactError(f"Invalid native configuration: {exc}") from exc
    if not type(config).__module__.startswith("transformers."):
        raise UnsupportedModelError("Third-party config implementations are outside the target scope")
    return config


def select_auto_class(config: PretrainedConfig) -> tuple[type, list[str]]:
    architectures = config.architectures or []
    for name in sorted(n for n in dir(transformers) if n.startswith("AutoModel")):
        auto_class = getattr(transformers, name)
        mapping = getattr(auto_class, "_model_mapping", None)
        if mapping is None or type(config) not in mapping:
            continue
        try:
            models = mapping[type(config)]
        except (ImportError, RuntimeError) as exc:
            raise UnsupportedModelError(f"Native architecture dependencies are unavailable: {exc}") from exc
        if not isinstance(models, tuple):
            models = (models,)
        if any(model.__name__ in architectures for model in models):
            return auto_class, []
    if type(config) not in AutoModel._model_mapping:
        raise UnsupportedModelError(f"No native PyTorch AutoModel for {type(config).__name__}")
    return AutoModel, [
        "No registered task wrapper matched config.architectures; selected the native AutoModel backbone"
    ]


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
    except (ImportError, RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
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
