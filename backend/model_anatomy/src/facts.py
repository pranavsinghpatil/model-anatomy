"""Deterministic module, parameter, buffer, and source facts without interpretation."""

from __future__ import annotations

import hashlib
import inspect
import re
from typing import Any

import torch
import transformers

from . import limits
from .artifacts import CheckpointMetadata, InspectionBudgetError, SafetensorsMetadata
from .errors import ArtifactError, NativeShapeMismatchError

ATTRIBUTES = (
    "training",
    "in_features",
    "out_features",
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "normalized_shape",
    "eps",
    "p",
    "num_embeddings",
    "embedding_dim",
    "num_heads",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "is_decoder",
    "is_cross_attention",
    "is_causal",
    "has_relative_attention_bias",
    "scaling",
    "output_dim",
    "nf",
    "n_heads",
    "key_value_proj_dim",
    "num_key_value_groups",
    "attention_type",
    "layer_type",
    "layer_idx",
    "sliding_window",
    "window_size",
    "local_attention",
    "global_attn_every_n_layers",
)


def inspect_native_model(
    model: torch.nn.Module, checkpoint: CheckpointMetadata | SafetensorsMetadata
) -> dict[str, Any]:
    if len(checkpoint.tensors) > limits.MAX_CHECKPOINT_TENSORS:
        raise InspectionBudgetError("Checkpoint exceeds the supported tensor record count")
    module_items = []
    for item in model.named_modules(remove_duplicate=False):
        if len(module_items) >= limits.MAX_NATIVE_MODULE_RECORDS:
            raise InspectionBudgetError("Native model exceeds the supported module record count")
        module_items.append(item)
    named_items = []
    for item in model.named_parameters(remove_duplicate=False):
        if len(named_items) >= limits.MAX_NATIVE_PARAMETER_RECORDS:
            raise InspectionBudgetError("Native model exceeds the supported parameter record count")
        named_items.append(item)
    buffer_items = []
    for item in model.named_buffers(remove_duplicate=False):
        if len(buffer_items) >= limits.MAX_NATIVE_BUFFER_RECORDS:
            raise InspectionBudgetError("Native model exceeds the supported buffer record count")
        buffer_items.append(item)
    modules: dict[str, dict[str, Any]] = {}
    source_types: dict[str, dict[str, Any]] = {}
    fingerprints = {}
    for path, module in module_items:
        qualified = f"{type(module).__module__}.{type(module).__name__}"
        attributes = {}
        for name in ATTRIBUTES:
            value = vars(module).get(name, inspect.getattr_static(module, name, None))
            if type(value) in (int, float, bool, str) or isinstance(value, (tuple, list, torch.Size)):
                attributes[name] = list(value) if isinstance(value, (tuple, list, torch.Size)) else value
        modules[path] = {
            "path": path,
            "type": type(module).__name__,
            "qualified_type": qualified,
            "parent": path.rpartition(".")[0] if path else None,
            "children": [f"{path}.{name}" if path else name for name, child in module._modules.items() if child is not None],
            "parameters": [
                f"{path}.{name}" if path else name for name, _ in module.named_parameters(recurse=False)
            ],
            "attributes": attributes,
            "buffers": [
                f"{path}.{name}" if path else name for name, _ in module.named_buffers(recurse=False)
            ],
            "container": "sequential"
            if isinstance(module, torch.nn.Sequential)
            else "module_list"
            if isinstance(module, torch.nn.ModuleList)
            else "module_dict"
            if isinstance(module, torch.nn.ModuleDict)
            else None,
        }
        if isinstance(module, torch.nn.Dropout):
            modules[path]["dropout_probability"] = module.p
        if isinstance(module, torch.nn.LayerNorm):
            modules[path]["epsilon"] = module.eps
        if qualified not in source_types:
            try:
                text = inspect.getsource(type(module).forward)
                source_types[qualified] = {
                    "qualified_name": qualified + ".forward",
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            except (OSError, TypeError):
                source_types[qualified] = {"qualified_name": qualified + ".forward", "sha256": None}
        fingerprints[path] = source_types[qualified]
    named = dict(named_items)
    prefix = model.base_model_prefix + "." if model.base_model_prefix else ""
    renamings = []
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming

        renamings = [rule for rule in get_model_conversion_mapping(model) if isinstance(rule, WeightRenaming)]
    except ImportError:
        pass
    checkpoint_names: dict[str, str] = {}
    name_conversions = {}
    legacy_normalizer = getattr(model, "_fix_state_dict_key_on_load", None)
    normalizer_evidence = None
    if inspect.isfunction(legacy_normalizer) and legacy_normalizer.__module__.startswith("transformers."):
        normalizer_evidence = {
            "qualified_name": f"{legacy_normalizer.__module__}.{legacy_normalizer.__qualname__}",
            "source_sha256": hashlib.sha256(inspect.getsource(legacy_normalizer).encode()).hexdigest(),
            "scope": "Native string-only checkpoint key normalization; no tensors loaded",
        }
    for name in checkpoint.tensors:
        converted = name
        steps = []
        if normalizer_evidence is not None:
            converted, _ = legacy_normalizer(converted)
            if not isinstance(converted, str) or len(converted) > 4096:
                raise ArtifactError("Native checkpoint key normalization returned an invalid name")
            if converted != name:
                steps.append(dict(normalizer_evidence, target=converted))
        for rule in renamings:
            converted, _ = rule.rename_source_key(converted)
        if converted != name:
            name_conversions[name] = {
                "target": converted,
                "source": "native Transformers checkpoint loading and conversion rules",
                "steps": steps,
            }
        candidates = {converted, prefix + converted, converted.removeprefix(prefix)} & named.keys()
        # Shared embedding paths can legitimately provide more than one candidate.
        if len(candidates) > 1 and len({id(named[p]) for p in candidates}) > 1:
            raise ArtifactError(f"Ambiguous checkpoint/module naming for {name}")
        for module_name in candidates:
            if module_name in checkpoint_names and checkpoint_names[module_name] != name:
                raise ArtifactError(f"Ambiguous checkpoint naming for {module_name}")
            checkpoint_names[module_name] = name
    aliases: dict[int, list[str]] = {}
    for name, parameter in named.items():
        aliases.setdefault(id(parameter), []).append(name)
    warnings = []
    parameters = {}
    for name, parameter in named.items():
        shared = aliases[id(parameter)]
        stored = sorted({checkpoint_names[alias] for alias in shared if alias in checkpoint_names})
        for tensor_name in stored:
            if tuple(parameter.shape) != checkpoint.tensors[tensor_name].shape:
                raise NativeShapeMismatchError(f"Checkpoint/module shape mismatch: {tensor_name}")
        if name == shared[0]:
            if not stored:
                warnings.append(f"No checkpoint tensor matched model parameter {name}")
            elif len(stored) > 1:
                warnings.append(
                    f"Model shares {shared}, but checkpoint stores multiple tensors; equality of their values is unverified"
                )
        tensor_name = checkpoint_names.get(name, stored[0] if stored else None)
        parameters[name] = {
            "name": name,
            "shape": list(parameter.shape),
            "parameter_count": parameter.numel(),
            "device": parameter.device.type,
            "module_dtype": str(parameter.dtype),
            "dtype": checkpoint.tensors[tensor_name].dtype if tensor_name else None,
            "tensor_name": tensor_name,
            "tensor_names": stored,
            "aliases": shared,
            "canonical_parameter": shared[0],
            "checkpoint_matched": bool(stored),
        }
    buffers = {
        name: {"shape": list(buffer.shape), "dtype": str(buffer.dtype), "device": buffer.device.type}
        for name, buffer in buffer_items
    }
    extras = {}
    native_buffers = dict(buffer_items)
    for name, tensor in checkpoint.tensors.items():
        if name in checkpoint_names.values():
            continue
        converted = name_conversions.get(name, {}).get("target", name)
        candidates = {converted, prefix + converted, converted.removeprefix(prefix)} & buffers.keys()
        owners = [p for p in candidates if tuple(buffers[p]["shape"]) == tensor.shape]
        if len({id(native_buffers[path]) for path in owners}) > 1:
            raise ArtifactError(f"Ambiguous checkpoint/buffer naming for {name}")
        if owners:
            extras[name] = {"kind": "buffer", "evidence_kind": "reconstructed", "buffer_paths": owners,
                            "checkpoint_dtype": tensor.dtype}
        elif any(re.search(pattern, name) for pattern in (model._keys_to_ignore_on_load_unexpected or [])):
            extras[name] = {
                "kind": "legacy_checkpoint_tensor",
                "evidence_kind": "reconstructed",
                "reason": "Native model declares this key ignored on checkpoint loading",
            }
        else:
            extras[name] = {
                "kind": "unmapped",
                "evidence_kind": "unknown",
                "reason": "No native parameter or buffer matched this checkpoint tensor",
            }
    return {
        "model_class": type(model).__name__,
        "base_model_prefix": model.base_model_prefix,
        "modules": modules,
        "parameters": parameters,
        "buffers": buffers,
        "checkpoint_extras": extras,
        "source_fingerprints": fingerprints,
        "source_types": source_types,
        "checkpoint_name_mapping": checkpoint_names,
        "checkpoint_name_conversions": name_conversions,
        "unique_parameter_count": sum(p.numel() for p in model.parameters()),
        "logical_parameter_count": sum(p.numel() for p in named.values()),
        "parameter_alias_groups": [group for group in aliases.values() if len(group) > 1],
        "execution": {"status": "not_run", "reason": "Forward execution is disabled by user instruction"},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "warnings": warnings,
    }
