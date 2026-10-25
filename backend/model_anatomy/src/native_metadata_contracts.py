"""Source-fingerprinted access to selected native metadata and Torch descriptors.

The helpers inspect registered meta parameters and static descriptors without
invoking model forwards, descriptor getters, or publisher code.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig, activations
from transformers.modeling_utils import ModuleUtilsMixin

CONTRACT_SOURCE_SHA256 = {
    "transformers.configuration_utils": "08a3889c1b8a73d340a37c93f72346ee0e385e339b7a72a03b5c026d4943baf7",
    "transformers.modeling_utils": "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
}

_MODULE_UTILS_DTYPE_PROPERTY = ModuleUtilsMixin.dtype
_MODULE_UTILS_DTYPE_GETTER = ModuleUtilsMixin.dtype.fget
_DTYPE_BY_TORCH = {
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
_TORCH_TENSOR_METHODS = {
    name: inspect.getattr_static(torch.Tensor, name)
    for name in ("clone", "expand", "masked_fill", "to", "__getitem__", "__gt__", "__eq__", "__mul__", "__imul__", "__add__", "__rsub__", "__setitem__")
}
_NN_MODULE_TRAVERSAL = {
    name: inspect.getattr_static(nn.Module, name)
    for name in ("parameters", "named_parameters", "_named_members", "named_modules")
}


def native_config_attribute(config: object, name: str) -> tuple[bool, Any]:
    """Read the reviewed attention implementation property without invoking it."""
    if type(name) is not str or name != "_attn_implementation":
        return False, None
    descriptor = inspect.getattr_static(type(config), name, None)
    trusted = inspect.getattr_static(PretrainedConfig, name, None)
    if (
        not isinstance(config, PretrainedConfig)
        or not isinstance(descriptor, property)
        or not isinstance(trusted, property)
        or descriptor.fget is not trusted.fget
    ):
        return False, None
    try:
        source = inspect.getsourcefile(trusted.fget)
        if (
            not source
            or hashlib.sha256(Path(source).read_bytes()).hexdigest()
            != CONTRACT_SOURCE_SHA256["transformers.configuration_utils"]
        ):
            return False, None
    except (OSError, TypeError):
        return False, None
    backing = vars(config).get("_attn_implementation_internal")
    if type(backing) is str and backing:
        return True, backing
    return False, None


def native_module_attribute(module: object, name: str) -> tuple[bool, Any]:
    """Read the reviewed T5 stack dtype from meta parameter metadata only."""
    if type(name) is not str or name != "dtype":
        return False, None

    from transformers.models.t5.modeling_t5 import T5Stack

    descriptor = inspect.getattr_static(type(module), name, None)
    if (
        type(module) is not T5Stack
        or descriptor is not _MODULE_UTILS_DTYPE_PROPERTY
        or _MODULE_UTILS_DTYPE_GETTER.__module__ != "transformers.modeling_utils"
        or _MODULE_UTILS_DTYPE_GETTER.__qualname__ != "ModuleUtilsMixin.dtype"
        or not _trusted_modeling_utils_source()
        or any(
            os.environ.get(flag, "").upper() in {"1", "ON", "TRUE", "YES"}
            for flag in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16")
        )
        or any(
            _MODULE_UTILS_DTYPE_GETTER.__globals__.get(flag) in {"1", "ON", "TRUE", "YES"}
            for flag in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16")
        )
        or not _native_module_tree(module)
    ):
        return False, None

    dtypes = _registered_meta_parameter_dtypes(module)
    if not dtypes or len(set(dtypes)) != 1:
        return False, None
    return True, next(dtype for dtype, name in _DTYPE_BY_TORCH.items() if name == dtypes[0])


def trusted_t5_stack_method(module: object, method_name: str, method: object) -> bool:
    """Validate the exact source function used to justify a T5 tensor fact."""
    from transformers.models.t5.modeling_t5 import T5Stack

    if type(module) is not T5Stack or method_name not in {
        "forward",
        "_prepare_4d_causal_attention_mask_with_cache_position",
    }:
        return False
    if "forward" in vars(module) or module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
        return False
    try:
        raw_descriptor = inspect.getattr_static(T5Stack, method_name)
        raw_method = raw_descriptor.__func__ if isinstance(raw_descriptor, staticmethod) else raw_descriptor
        function = inspect.unwrap(raw_method)
        source = inspect.getsourcefile(function)
        return (
            function is inspect.unwrap(method)
            and function.__module__ == "transformers.models.t5.modeling_t5"
            and function.__qualname__ == f"T5Stack.{method_name}"
            and bool(source)
            and hashlib.sha256(Path(source).read_bytes()).hexdigest()
            == _t5_source_hash()
        )
    except (AttributeError, OSError, TypeError):
        return False


def trusted_t5_module_utils_helper(module: object, method_name: str, method: object) -> bool:
    """Trust the exact inherited mask conversion helper on a native T5 stack."""
    from transformers.models.t5.modeling_t5 import T5Stack

    if type(module) is not T5Stack or method_name != "invert_attention_mask" or method_name in vars(module):
        return False
    if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
        return False
    try:
        descriptor = inspect.getattr_static(T5Stack, method_name)
        function = inspect.unwrap(descriptor)
        source = inspect.getsourcefile(function)
        return (
            descriptor is ModuleUtilsMixin.invert_attention_mask
            and function is inspect.unwrap(method)
            and function.__module__ == "transformers.modeling_utils"
            and function.__qualname__ == "ModuleUtilsMixin.invert_attention_mask"
            and bool(source)
            and hashlib.sha256(Path(source).read_bytes()).hexdigest()
            == CONTRACT_SOURCE_SHA256["transformers.modeling_utils"]
        )
    except (AttributeError, OSError, TypeError, KeyError):
        return False


def trusted_torch_tensor_method(name: str) -> bool:
    """Reject runtime replacement of the Torch tensor methods used by T5 facts."""
    expected = _TORCH_TENSOR_METHODS.get(name)
    return expected is not None and inspect.getattr_static(torch.Tensor, name, None) is expected


def _t5_source_hash() -> str:
    # Import lazily to keep this metadata module independent of graph construction.
    from .native_output_contracts import CONTRACT_SOURCE_SHA256 as output_sources

    return output_sources["transformers.models.t5.modeling_t5"]


def _trusted_modeling_utils_source() -> bool:
    try:
        source = inspect.getsourcefile(_MODULE_UTILS_DTYPE_GETTER)
        return (
            bool(source)
            and hashlib.sha256(Path(source).read_bytes()).hexdigest()
            == CONTRACT_SOURCE_SHA256["transformers.modeling_utils"]
            and _MODULE_UTILS_DTYPE_GETTER is ModuleUtilsMixin.dtype.fget
            and _MODULE_UTILS_DTYPE_GETTER.__module__ == "transformers.modeling_utils"
            and _MODULE_UTILS_DTYPE_GETTER.__qualname__ == "ModuleUtilsMixin.dtype"
        )
    except (OSError, TypeError, KeyError):
        return False


def _native_module_tree(module: nn.Module) -> bool:
    """Require the standard nn.Module traversal over native T5/torch children."""
    from transformers.models.t5 import modeling_t5 as t5

    pending = [module]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen or len(seen) >= 4096 or type(current) not in _allowed_t5_module_types(t5):
            return False
        seen.add(id(current))
        for name in ("parameters", "named_parameters", "_named_members", "named_modules"):
            if name in vars(current) or inspect.getattr_static(type(current), name, None) is not _NN_MODULE_TRAVERSAL[name]:
                return False
        if not isinstance(current._modules, dict) or not isinstance(current._parameters, dict):
            return False
        if any(parameter is not None and not isinstance(parameter, nn.Parameter) for parameter in current._parameters.values()):
            return False
        if any(child is not None and not isinstance(child, nn.Module) for child in current._modules.values()):
            return False
        pending.extend(child for child in current._modules.values() if child is not None)
    return True


def _allowed_t5_module_types(t5: Any) -> set[type]:
    return {
        nn.Module,
        nn.ModuleList,
        nn.Embedding,
        nn.Dropout,
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        activations.GELUActivation,
        activations.NewGELUActivation,
        activations.SiLUActivation,
        t5.T5Stack,
        t5.T5Block,
        t5.T5Attention,
        t5.T5LayerSelfAttention,
        t5.T5LayerCrossAttention,
        t5.T5LayerNorm,
        t5.T5LayerFF,
        t5.T5DenseActDense,
        t5.T5DenseGatedActDense,
    }


def _registered_meta_parameter_dtypes(module: nn.Module) -> list[str]:
    result: list[str] = []
    pending = [module]
    while pending:
        current = pending.pop()
        for parameter in current._parameters.values():
            if parameter is None:
                continue
            dtype = _DTYPE_BY_TORCH.get(parameter.dtype)
            if parameter.device.type != "meta" or dtype is None:
                return []
            result.append(dtype)
        pending.extend(reversed([child for child in current._modules.values() if child is not None]))
    return result
