"""Inspect a real Transformers model on meta and reconcile checkpoint metadata."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from .artifacts import SafetensorsMetadata
from .errors import ArtifactError, UnsupportedModelError


@dataclass(frozen=True)
class GPT2Dimensions:
    vocabulary: int
    context: int
    hidden: int
    layers: int
    heads: int
    inner: int
    activation: str

    @property
    def head_dimension(self) -> int:
        return self.hidden // self.heads


def load_config(data: dict[str, Any]) -> tuple[PretrainedConfig, GPT2Dimensions]:
    if data.get("model_type") != "gpt2" or data.get("auto_map"):
        raise UnsupportedModelError("Only built-in GPT-2 configuration is supported; remote code is disabled")
    if data.get("architectures") != ["GPT2LMHeadModel"]:
        raise UnsupportedModelError("Expected GPT2LMHeadModel architecture")
    for key in ("vocab_size", "n_positions", "n_embd", "n_layer", "n_head"):
        if type(data.get(key)) is not int or data[key] <= 0:
            raise ArtifactError(f"Missing or invalid GPT-2 configuration field: {key}")
    config = AutoConfig.for_model("gpt2", **{k: v for k, v in data.items() if k != "model_type"})
    if (
        config.add_cross_attention
        or not config.tie_word_embeddings
        or not config.scale_attn_weights
        or config.scale_attn_by_inverse_layer_idx
        or config.reorder_and_upcast_attn
        or data.get("pruned_heads")
    ):
        raise UnsupportedModelError(
            "Cross-attention, untied weights, pruning, and alternate attention scaling are unsupported"
        )
    inner = config.n_inner if config.n_inner is not None else 4 * config.n_embd
    if config.n_embd % config.n_head or type(inner) is not int or inner <= 0:
        raise ArtifactError("Invalid attention-head or feed-forward dimensions")
    if config.activation_function != "gelu_new":
        raise UnsupportedModelError("The validated GPT-2 semantic rules require gelu_new")
    return config, GPT2Dimensions(
        config.vocab_size,
        config.n_positions,
        config.n_embd,
        config.n_layer,
        config.n_head,
        inner,
        config.activation_function,
    )


def instantiate(config: PretrainedConfig) -> torch.nn.Module:
    """Construct metadata tensors only; never call from_pretrained on a model."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, attn_implementation="eager")
    model.tie_weights()
    model.eval()
    if any(t.device.type != "meta" for t in (*model.parameters(), *model.buffers())):
        raise ArtifactError("Model construction unexpectedly allocated non-meta tensors")
    return model


def inspect_model(model: torch.nn.Module, checkpoint: SafetensorsMetadata) -> dict[str, Any]:
    modules: dict[str, dict[str, Any]] = {}
    for path, module in model.named_modules():
        modules[path] = {
            "path": path,
            "type": type(module).__name__,
            "qualified_type": f"{type(module).__module__}.{type(module).__name__}",
            "parent": path.rpartition(".")[0] if path else None,
            "children": [f"{path}.{name}" if path else name for name, _ in module.named_children()],
            "parameters": [
                f"{path}.{name}" if path else name for name, _ in module.named_parameters(recurse=False)
            ],
        }
        if isinstance(module, torch.nn.Dropout):
            modules[path]["dropout_probability"] = module.p
        if isinstance(module, torch.nn.LayerNorm):
            modules[path]["epsilon"] = module.eps
    named = dict(model.named_parameters(remove_duplicate=False))
    base_prefix = model.base_model_prefix + "."
    checkpoint_names = {}
    for name in checkpoint.tensors:
        module_name = name if name in named else base_prefix + name
        if module_name in named:
            if module_name in checkpoint_names:
                raise ArtifactError(f"Ambiguous checkpoint naming for {module_name}")
            checkpoint_names[module_name] = name
    aliases: dict[int, list[str]] = {}
    for name, parameter in named.items():
        aliases.setdefault(id(parameter), []).append(name)
    parameters: dict[str, dict[str, Any]] = {}
    for name, parameter in named.items():
        shared = aliases[id(parameter)]
        stored = [checkpoint_names[alias] for alias in shared if alias in checkpoint_names]
        if not stored:
            raise ArtifactError(f"No checkpoint tensor for model parameter: {name}")
        if len(stored) > 1:
            raise ArtifactError(
                f"Multiple stored tensors for tied parameter {name}; equality cannot be verified without weight values"
            )
        for tensor_name in stored:
            if tuple(parameter.shape) != checkpoint.tensors[tensor_name].shape:
                raise ArtifactError(f"Checkpoint/module shape mismatch: {tensor_name}")
            if checkpoint.tensors[tensor_name].dtype != "F32":
                raise UnsupportedModelError(f"The GPT-2 MVP validates F32 parameters; got {tensor_name}")
        tensor_name = min(stored)
        parameters[name] = {
            "name": name,
            "shape": list(parameter.shape),
            "parameter_count": parameter.numel(),
            "device": parameter.device.type,
            "module_dtype": str(parameter.dtype),
            "dtype": checkpoint.tensors[tensor_name].dtype,
            "tensor_name": tensor_name,
            "aliases": shared,
            "canonical_parameter": shared[0],
        }
    buffers = {
        name: {"shape": list(buffer.shape), "dtype": str(buffer.dtype), "device": buffer.device.type}
        for name, buffer in model.named_buffers()
    }
    extras = {}
    mapped_tensors = set(checkpoint_names.values())
    for name, tensor in checkpoint.tensors.items():
        if name in mapped_tensors:
            continue
        # Historical GPT-2 checkpoints include generated causal-mask buffers.
        module_name = name if name in buffers else base_prefix + name
        if module_name in buffers and tuple(buffers[module_name]["shape"]) == tensor.shape:
            extras[name] = {"kind": "buffer", "reason": "matched named_buffers by name and shape"}
        elif (
            module_name in {f"transformer.h.{i}.attn.bias" for i in range(model.config.n_layer)}
            and tensor.shape == (1, 1, model.config.n_positions, model.config.n_positions)
            and tensor.dtype in {"BOOL", "U8", "F32"}
        ):
            extras[name] = {
                "kind": "legacy_buffer",
                "evidence_kind": "reconstructed",
                "module_path": module_name.rpartition(".")[0],
                "reason": "GPT-2 historical causal-mask name and shape; values not inspected",
            }
        else:
            raise ArtifactError(f"Checkpoint tensor has no module parameter or buffer owner: {name}")
    source = {}
    for path in ("", "transformer", "transformer.h.0", "transformer.h.0.attn", "transformer.h.0.mlp"):
        module = model.get_submodule(path) if path else model
        method = type(module).forward
        text = inspect.getsource(method)
        source[path] = {
            "qualified_name": f"{type(module).__module__}.{type(module).__name__}.forward",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    attention_type = type(model.get_submodule("transformer.h.0.attn"))
    attention_module = inspect.getmodule(attention_type)
    kernel = getattr(attention_module, "eager_attention_forward", None)
    if kernel is None:
        kernel = getattr(attention_type, "_attn", None)
    if kernel is None:
        raise UnsupportedModelError("Cannot identify the installed GPT-2 eager attention source")
    source["attention_kernel"] = {
        "qualified_name": f"{kernel.__module__}.{kernel.__qualname__}",
        "sha256": hashlib.sha256(inspect.getsource(kernel).encode()).hexdigest(),
    }
    return {
        "modules": modules,
        "parameters": parameters,
        "buffers": buffers,
        "checkpoint_extras": extras,
        "source_fingerprints": source,
        "checkpoint_name_mapping": checkpoint_names,
        "unique_parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_alias_groups": [group for group in aliases.values() if len(group) > 1],
        "execution": {"status": "not_run", "reason": "Metadata-only meta tensors have no weight values"},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }


def detect_repetition(raw: dict[str, Any], dimensions: GPT2Dimensions) -> list[str]:
    modules, parameters = raw["modules"], raw["parameters"]
    paths = modules.get("transformer.h", {}).get("children", [])
    expected = [f"transformer.h.{i}" for i in range(dimensions.layers)]
    if paths != expected:
        raise ArtifactError("Configured layer count disagrees with discovered transformer blocks")

    def signature(prefix: str) -> tuple:
        module_entries = [
            (p[len(prefix) :], m["qualified_type"], m.get("dropout_probability"), m.get("epsilon"))
            for p, m in modules.items()
            if p == prefix or p.startswith(prefix + ".")
        ]
        parameter_entries = [
            (p[len(prefix) :], tuple(v["shape"]), v["dtype"])
            for p, v in parameters.items()
            if p.startswith(prefix + ".")
        ]
        return tuple(module_entries), tuple(parameter_entries)

    first = signature(paths[0])
    if any(modules[p]["type"] != "GPT2Block" or signature(p) != first for p in paths):
        raise ArtifactError("Transformer blocks are not structurally equivalent; cannot collapse them")
    return paths
