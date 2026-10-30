"""Use native Transformers Auto mappings instead of maintaining a model registry."""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from typing import Any

import torch
from transformers import AutoConfig, PretrainedConfig

from . import limits
from .errors import ArtifactError, UnsupportedModelError
from .native_provider import (
    NativeResolutionError,
    config_class,
    implementation_evidence,
    model_candidates,
)

_CONSTRUCTING = ContextVar("model_anatomy_meta_construction", default=False)


def _forbid_forward(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
    if _CONSTRUCTING.get():
        raise UnsupportedModelError(
            "Native construction attempted forward execution; it is disabled"
        )


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
_STAGE_DEPTH_MODEL_TYPES = {
    "bit",
    "swin",
    "swinv2",
    "donut-swin",
    "maskformer-swin",
    "segformer",
}
_IMPLICIT_BACKBONE_STAGE_TYPES = {
    "dpt": "bit",
    "mask2former": "swin",
    "maskformer": "swin",
    "oneformer": "swin",
    "vit-hybrid": "bit",
}
_IGNORED_AUTO_MAP_WARNING = (
    "Publisher root auto_map was ignored because every declared architecture matched an installed native "
    "Transformers mapping; equivalence to publisher custom code is unverified, and remote code remains disabled"
)
_IGNORED_AUTO_MAP_MARKER = "_model_anatomy_ignored_publisher_auto_map"


def _clap_audio_schedule_defaults() -> dict[str, Any]:
    """Read schedule defaults from the installed native config without constructing it."""
    from inspect import signature

    from transformers.models.clap.configuration_clap import ClapAudioConfig

    parameters = signature(ClapAudioConfig.__init__).parameters
    return {
        name: (
            list(parameters[name].default)
            if isinstance(parameters[name].default, tuple)
            else parameters[name].default
        )
        for name in ("depths", "num_hidden_layers", "num_attention_heads")
    }


def _validate_embedded_config(
    data: dict[str, Any],
    *,
    allow_root_auto_map: bool = False,
    validate_native_fields: bool = True,
) -> None:
    if not isinstance(data, dict):
        raise ArtifactError("Native configuration must be an object")
    pending: list[tuple[Any, int, str | None, bool]] = [(data, 0, None, True)]
    visited = 0
    while pending:
        value, depth, inherited_stage_type, construction_config = pending.pop()
        if not isinstance(value, (dict, list)):
            continue
        visited += 1
        if visited > limits.MAX_CONFIG_CONTAINER_NODES:
            raise ArtifactError("Configuration exceeds the nested metadata node limit")
        if depth > limits.MAX_CONFIG_NESTING_DEPTH:
            raise ArtifactError("Configuration exceeds the nested metadata depth limit")
        if isinstance(value, list):
            pending.extend(
                (item, depth + 1, None, False)
                for item in value
                if isinstance(item, (dict, list))
            )
            continue
        if not validate_native_fields:
            pending.extend(
                (item, depth + 1, None, False)
                for item in value.values()
                if isinstance(item, (dict, list))
            )
            continue
        if value.get("auto_map") and not (allow_root_auto_map and depth == 0):
            raise UnsupportedModelError(
                "Native configuration is required; remote code is disabled"
            )

        construction_config = construction_config or isinstance(
            value.get("model_type"), str
        )
        if construction_config and (
            value.get("use_pretrained_backbone") or value.get("pretrained")
        ):
            raise UnsupportedModelError(
                "Pretrained backbone initialization is disabled; weights must not be loaded"
            )
        if (
            construction_config
            and value.get("backbone")
            and value.get("backbone_config") is None
        ):
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
                    for canonical, alias in CONFIG_MAPPING[
                        model_type
                    ].attribute_map.items()
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
        if stage_type == "segformer":
            num_encoder_blocks = value.get("num_encoder_blocks", 4)
            if type(num_encoder_blocks) is not int or num_encoder_blocks <= 0:
                raise ArtifactError(
                    "Invalid configuration dimension: num_encoder_blocks"
                )
            if num_encoder_blocks > limits.MAX_MODEL_LAYERS:
                raise UnsupportedModelError(
                    f"Model exceeds the supported layer count ({limits.MAX_MODEL_LAYERS})"
                )
            if "depths" in value and len(value["depths"]) != num_encoder_blocks:
                raise ArtifactError("Invalid SegFormer stage schedule: depths")
            heads = value.get("num_attention_heads")
            if heads is not None and (
                not isinstance(heads, list)
                or len(heads) != num_encoder_blocks
                or any(type(head) is not int or head <= 0 for head in heads)
            ):
                raise ArtifactError(
                    "Invalid SegFormer stage schedule: num_attention_heads"
                )
        if stage_type == "clap_audio_model":
            defaults = _clap_audio_schedule_defaults()
            depths = value.get("depths", defaults["depths"])
            if (
                not isinstance(depths, list)
                or not depths
                or any(type(n) is not int or n <= 0 for n in depths)
            ):
                raise ArtifactError("Invalid CLAP audio stage schedule: depths")
            if sum(depths) > limits.MAX_MODEL_LAYERS:
                raise UnsupportedModelError(
                    f"Model exceeds the supported layer count ({limits.MAX_MODEL_LAYERS})"
                )
            stage_count = value.get("num_hidden_layers", defaults["num_hidden_layers"])
            if (
                type(stage_count) is not int
                or stage_count <= 0
                or stage_count != len(depths)
            ):
                raise ArtifactError(
                    "Invalid CLAP audio stage schedule: num_hidden_layers"
                )
            if stage_count > limits.MAX_MODEL_LAYERS:
                raise UnsupportedModelError(
                    f"Model exceeds the supported layer count ({limits.MAX_MODEL_LAYERS})"
                )
            heads = value.get("num_attention_heads", defaults["num_attention_heads"])
            if (
                not isinstance(heads, list)
                or len(heads) != len(depths)
                or any(
                    type(n) is not int or n <= 0 or n > limits.MAX_MODEL_LAYERS
                    for n in heads
                )
            ):
                raise ArtifactError(
                    "Invalid CLAP audio stage schedule: num_attention_heads"
                )
        for name in _DIMENSION_FIELDS | layer_fields:
            if name not in value:
                continue
            if (
                name == "num_attention_heads"
                and stage_type == "segformer"
                and isinstance(value[name], list)
            ):
                continue
            if name == "num_attention_heads" and stage_type == "clap_audio_model":
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
                construction_config
                and (name.endswith("_config") or name in {"encoder", "decoder"}),
            )
            for name, item in value.items()
            if isinstance(item, (dict, list))
        )


def load_native_config(data: dict[str, Any]) -> PretrainedConfig:
    if not isinstance(data, dict):
        raise ArtifactError("Native configuration must be an object")
    # Artifact-only reports retain the original config too. Apply structural
    # budgets before any unsupported-model decision can return that metadata.
    _validate_embedded_config(data, validate_native_fields=False)
    model_type = data.get("model_type")
    if not isinstance(model_type, str):
        raise UnsupportedModelError(
            "A native Transformers config is required; remote code is disabled"
        )
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type not in CONFIG_MAPPING and model_type in _IMPLICIT_BACKBONE_STAGE_TYPES:
        _validate_embedded_config(data)
    implementation = config_class(model_type)
    publisher_auto_map = data.get("auto_map")
    if publisher_auto_map is not None and not isinstance(publisher_auto_map, dict):
        raise ArtifactError("Invalid publisher auto_map; expected an object")
    ignored_auto_map = bool(publisher_auto_map)
    declared_architectures = data.get("architectures")
    native_data = {
        key: value for key, value in data.items()
        if key != _IGNORED_AUTO_MAP_MARKER
    }
    if ignored_auto_map:
        if (
            not isinstance(declared_architectures, list)
            or not declared_architectures
            or any(not isinstance(name, str) or not name for name in declared_architectures)
        ):
            raise UnsupportedModelError(
                "Publisher auto_map can be ignored only with nonempty declared native architectures"
            )
        native_data.pop("auto_map", None)
    elif publisher_auto_map is None:
        native_data.pop("auto_map", None)
    _validate_embedded_config(data, allow_root_auto_map=ignored_auto_map)
    pending = [(implementation, data, "config")]
    while pending:
        parent_class, values, path = pending.pop()
        for name in getattr(parent_class, "sub_configs", {}):
            nested = values.get(name)
            if isinstance(nested, dict) and isinstance(nested.get("model_type"), str):
                child = config_class(nested["model_type"], f"{path}.{name}")
                pending.append((child, nested, f"{path}.{name}"))
    try:
        # Native config normalization must not mutate the artifact JSON.
        native_data = deepcopy(native_data)
        config = AutoConfig.for_model(
            model_type, **{k: v for k, v in native_data.items() if k != "model_type"}
        )
    except ImportError as exc:
        raise NativeResolutionError(
            "native_dependency_missing", "config_resolution", str(exc)
        ) from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise ArtifactError(f"Invalid native configuration: {exc}") from exc
    except RecursionError as exc:
        raise ArtifactError("Native configuration is too deeply nested") from exc
    if not type(config).__module__.startswith("transformers."):
        raise UnsupportedModelError(
            "Third-party config implementations are outside the target scope"
        )
    if ignored_auto_map:
        native_architectures = {
            candidate[2] for candidate in _auto_model_candidates(config)
        }
        if any(name not in native_architectures for name in declared_architectures):
            raise UnsupportedModelError(
                "Publisher auto_map can be ignored only when every declared architecture matches an "
                "installed native Auto mapping"
            )
        setattr(config, _IGNORED_AUTO_MAP_MARKER, True)
    return config


def _selection_warnings(config: PretrainedConfig, warnings: list[str]) -> list[str]:
    if getattr(config, _IGNORED_AUTO_MAP_MARKER, False):
        return [_IGNORED_AUTO_MAP_WARNING, *warnings]
    return warnings


def select_auto_class(config: PretrainedConfig) -> tuple[type, list[str]]:
    architectures = config.architectures
    if architectures is None:
        architectures = []
    if not isinstance(architectures, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in architectures
    ):
        raise ArtifactError(
            "Invalid native configuration architectures; expected a list of class names"
        )

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
        return matches[0][1], _selection_warnings(config, [])

    if architectures:
        declared = ", ".join(architectures)
        raise NativeResolutionError(
            "native_model_missing",
            "model_resolution",
            f"No native AutoModel mapping for declared architecture(s): {declared}",
            declared_architectures=list(architectures),
            model_type=config.model_type,
        )

    hint = _task_routing_hint(config)
    if hint is not None:
        routed = next(
            (candidate for candidate in candidates if candidate[0] == hint), None
        )
        if routed is not None:
            return routed[1], [
                f"No architecture declared; selected {hint} from native config task metadata"
            ]

    backbone = next(
        (candidate for candidate in candidates if candidate[0] == "AutoModel"), None
    )
    if backbone is None:
        raise UnsupportedModelError(
            f"No native PyTorch AutoModel for {type(config).__name__}"
        )
    return backbone[1], [
        *_selection_warnings(config, []),
        "No architecture declared or compatible task wrapper found; selected the native AutoModel backbone",
    ]


def _auto_model_candidates(config: PretrainedConfig) -> list[tuple[str, type, str]]:
    return model_candidates(config)


def _task_routing_hint(config: PretrainedConfig) -> str | None:
    """Use generic native config modality flags only when architecture names do not decide."""
    if not getattr(config, "is_encoder_decoder", False):
        return None
    if getattr(config, "num_mel_bins", None) is not None:
        return "AutoModelForSpeechSeq2Seq"
    return "AutoModelForSeq2SeqLM"


def instantiate_native(
    config: PretrainedConfig,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    auto_class, warnings = select_auto_class(config)
    try:
        return _construct_native(
            config, auto_class.from_config, auto_class.__name__, warnings
        )
    except ImportError as exc:
        raise NativeResolutionError(
            "native_dependency_missing", "model_resolution", str(exc)
        ) from exc


def instantiate_candidate(
    config: PretrainedConfig, candidate: tuple[str, type, str]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Resolve one upstream implementation without rewriting the publisher config."""
    auto_name, auto_class, class_name = candidate
    mapping = auto_class._model_mapping
    implementations = mapping[type(config)]
    if not isinstance(implementations, (tuple, list)):
        implementations = (implementations,)
    matches = [
        implementation
        for implementation in implementations
        if implementation.__name__ == class_name
    ]
    if len(matches) != 1 or not matches[0].__module__.startswith("transformers."):
        raise UnsupportedModelError(
            "Candidate is not a unique native AutoModel implementation"
        )
    effective = deepcopy(config)
    effective.architectures = [class_name]
    model_class = matches[0]
    return _construct_native(
        effective, model_class._from_config, auto_name, [], concrete=True
    )


def _construct_native(
    config: PretrainedConfig,
    factory: Any,
    auto_name: str,
    warnings: list[str],
    *,
    concrete: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Keep every construction path under the same no-forward/meta-only guard."""
    # Eager implementation exposes inspectable source; no attention is executed.
    config._attn_implementation = "eager"
    token = _CONSTRUCTING.set(True)
    handle = torch.nn.modules.module.register_module_forward_pre_hook(_forbid_forward)
    from .construction_guards import construction_guards

    try:
        with construction_guards(), torch.device("meta"):
            model = (
                factory(config)
                if concrete
                else factory(config, trust_remote_code=False)
            )
            model.tie_weights()
            model.eval()
    except UnsupportedModelError:
        raise
    except (ValueError, TypeError) as exc:
        raise ArtifactError(
            f"Invalid native model configuration for {config.model_type}: {exc}"
        ) from exc
    except (ImportError, RuntimeError, NotImplementedError) as exc:
        code = (
            "native_dependency_missing"
            if isinstance(exc, ImportError)
            else "native_meta_construction_failed"
        )
        raise NativeResolutionError(
            code,
            "meta_construction",
            f"Native meta construction failed for {config.model_type}: {exc}",
            model_type=config.model_type,
            missing_package=getattr(exc, "name", None),
        ) from exc
    finally:
        handle.remove()
        _CONSTRUCTING.reset(token)
    if not type(model).__module__.startswith("transformers."):
        raise UnsupportedModelError(
            "Only native Transformers PyTorch models are supported"
        )
    if any(t.device.type != "meta" for t in (*model.parameters(), *model.buffers())):
        raise ArtifactError(
            "Native construction allocated non-meta model tensors; no CPU fallback is allowed"
        )
    return model, {
        "auto_class": auto_name,
        "model_class": type(model).__name__,
        "trust_remote_code": False,
        "device": "meta",
        "publisher_auto_map": (
            "ignored_native_mapping_selected"
            if getattr(config, _IGNORED_AUTO_MAP_MARKER, False)
            else "not_used"
        ),
        "warnings": warnings,
        "native_resolution": implementation_evidence(model),
    }


def construct_native(config: PretrainedConfig, auto_class: type) -> torch.nn.Module:
    """Construct a native model through the shared metadata-only guards."""
    model, _selection = _construct_native(
        config, auto_class.from_config, auto_class.__name__, []
    )
    return model


def native_auto_candidates(config: PretrainedConfig) -> list[tuple[str, type, str]]:
    return _auto_model_candidates(config)
