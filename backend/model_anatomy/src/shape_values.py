"""Serialize typed source returns and bind tensor leaves to explicit graph ports."""

from __future__ import annotations

from typing import Any

from .ir import TensorSpec
from .source_eval import Symbol, Tensor


def describe(value: Any) -> dict[str, Any]:
    if value.fields is not None:
        return {"kind": "model_output", "fields": {name: describe(item) for name, item in value.fields.items()}}
    if value.items is not None:
        return {"kind": "tuple", "items": [describe(item) for item in value.items]}
    if value.symbolic is not None:
        spec = value.symbolic.spec()
        return {"kind": "tensor", "shape": spec["shape"], "dtype": spec["dtype"]}
    if value.shape_spec is not None:
        return {"kind": "tensor", "shape": value.shape_spec.shape, "dtype": value.shape_spec.dtype}
    if value.constant is None:
        return {"kind": "none"}
    return {"kind": "unknown", "shape": None, "dtype": None}


def ports(structure: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    kind = structure.get("kind")
    if kind == "model_output":
        return {key: value for name, child in structure["fields"].items()
                for key, value in ports(child, f"{prefix}.{name}".strip(".")).items()}
    if kind == "tuple":
        return {key: value for index, child in enumerate(structure["items"])
                for key, value in ports(child, f"{prefix}.{index}".strip(".")).items()}
    if kind == "none":
        return {}
    return {prefix or "output": {"shape": structure.get("shape"), "dtype": structure.get("dtype")}}


def bind(structure: dict[str, Any], reference: str, prefix: str = "") -> Any:
    from .static_flow import StaticValue, combine

    kind = structure.get("kind")
    if kind == "model_output":
        fields = {name: bind(child, reference, f"{prefix}.{name}".strip(".")) for name, child in structure["fields"].items()}
        result = combine(list(fields.values()))
        result.fields, result.domain = fields, "model_output"
        result.items = [item for item in fields.values() if item.constant is not None]
        return result
    if kind == "tuple":
        items = [bind(child, reference, f"{prefix}.{index}".strip(".")) for index, child in enumerate(structure["items"])]
        result = combine(items)
        result.items, result.domain = items, "tuple"
        return result
    if kind == "none":
        return StaticValue.literal(None)
    spec = TensorSpec(structure.get("shape"), structure.get("dtype"), "source_output")
    symbolic = Tensor(tuple(Symbol(d) if isinstance(d, str) else d for d in spec.shape), spec.dtype, reference) if spec.shape is not None and spec.dtype else None
    return StaticValue([(reference, prefix or "output")], domain="tensor" if kind == "tensor" else "unknown", shape_spec=spec, symbolic=symbolic)
