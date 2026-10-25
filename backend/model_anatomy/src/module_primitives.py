"""Native basic-module contracts for unregistered functional expressions."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

import torch
from torch import nn

# These exact native implementations contain no registered-module callbacks.
# Subclasses and composite/stateful layers require ordinary source analysis.
_FUNCTIONAL_TYPES = frozenset({
    nn.Identity, nn.Flatten, nn.Unflatten,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.CELU, nn.SELU, nn.GELU,
    nn.SiLU, nn.Mish, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Softmin, nn.LogSoftmax,
    nn.Hardtanh, nn.Hardsigmoid, nn.Hardswish, nn.Softplus, nn.Softsign,
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d,
    nn.ZeroPad1d, nn.ZeroPad2d, nn.ZeroPad3d,
    nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReflectionPad3d,
    nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d,
    nn.CrossEntropyLoss, nn.NLLLoss, nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss,
    nn.BCELoss, nn.BCEWithLogitsLoss, nn.KLDivLoss,
})
_IMPLEMENTATIONS = {implementation: (inspect.getattr_static(implementation, "forward"), inspect.getattr_static(implementation, "__init__"))
                    for implementation in _FUNCTIONAL_TYPES}


def functional_primitive(value: Any) -> type | None:
    implementation = value if isinstance(value, type) else type(value)
    if implementation in _FUNCTIONAL_TYPES and (
            inspect.getattr_static(implementation, "forward"), inspect.getattr_static(implementation, "__init__")) == _IMPLEMENTATIONS[implementation] and (
            isinstance(value, type) or not value._modules and "forward" not in vars(value)):
        return implementation
    # Native Transformers activation objects are leaves with no child callbacks.
    if (isinstance(value, nn.Module) and type(value).__module__ == "transformers.activations"
            and not value._modules and not value._parameters and not value._buffers):
        return implementation
    return None


def script_primitive_proof(value: Any) -> dict[str, Any] | None:
    """Prove a native ScriptFunction contains only tensor/primitive operations."""
    if type(value) is not torch.jit.ScriptFunction:
        return None
    pending, count = [value.graph], 0
    while pending:
        block = pending.pop()
        for node in block.nodes():
            count += 1
            if count > 40_000 or not node.kind().startswith(("aten::", "prim::")):
                return None
            if node.kind() in {"prim::CallFunction", "prim::CallMethod", "prim::PythonOp", "prim::GetAttr", "prim::SetAttr", "prim::CreateObject"}:
                return None
            if any(output.type().kind() in {"ClassType", "InterfaceType", "AnyType", "PyObjectType"} for output in node.outputs()):
                return None
            pending.extend(node.blocks())
    if any(argument.type.kind() not in {"TensorType", "IntType", "FloatType", "BoolType"} for argument in value.schema.arguments):
        return None
    return {"rule": "native_script_tensor_operations", "graph_sha256": hashlib.sha256(str(value.graph).encode()).hexdigest(),
            "scope": "No module/object calls or Python callbacks in compiled metadata", "forward_executed": False}


def primitive_proof(implementation: type) -> dict[str, Any]:
    try:
        source_hash = hashlib.sha256(inspect.getsource(implementation.forward).encode()).hexdigest()
    except (OSError, TypeError):
        source_hash = None
    return {"implementation": f"{implementation.__module__}.{implementation.__qualname__}",
            "forward_source_sha256": source_hash,
            "basis": "Native basic-module implementation identity; no registered module callbacks",
            "scope": "Unregistered functional expression outside the registered-module graph",
            "forward_executed": False}
