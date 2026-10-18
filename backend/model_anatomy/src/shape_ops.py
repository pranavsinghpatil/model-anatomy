"""Bounded shape algebra for the canonical native ViT source path, never execution."""

from __future__ import annotations

from math import prod
from typing import Any

from .ir import TensorSpec


def operation_shape(kind: str, metadata: dict[str, Any], inputs: list[TensorSpec]) -> TensorSpec:
    shape = list(inputs[0].shape) if inputs and inputs[0].shape is not None else None
    dtype = inputs[0].dtype if inputs else None
    result = None
    if shape is not None:
        rank = len(shape)
        method = metadata.get("method")
        dimensions = metadata.get("dimensions", [])
        if kind == "selection":
            indices = metadata.get("indices", [])
            if len(indices) == rank and all(
                value is None or type(value) is int and type(size) is int and -size <= value < size
                for value, size in zip(indices, shape)
            ):
                result = [size for value, size in zip(indices, shape) if value is None]
        elif kind == "reshape" and not metadata.get("keyword_arguments"):
            if method == "flatten" and len(dimensions) in {0, 1, 2}:
                start = dimensions[0] if dimensions else 0
                end = dimensions[1] if len(dimensions) == 2 else -1
                if type(start) is int and type(end) is int and -rank <= start < rank and -rank <= end < rank:
                    start, end = start % rank, end % rank
                    if start <= end and all(type(v) is int for v in shape[start : end + 1]):
                        result = shape[:start] + [prod(shape[start : end + 1])] + shape[end + 1 :]
            elif (
                method == "transpose"
                and len(dimensions) == 2
                and all(type(v) is int and -rank <= v < rank for v in dimensions)
            ):
                a, b = (v % rank for v in dimensions)
                result = list(shape)
                result[a], result[b] = result[b], result[a]
            elif method == "expand" and len(dimensions) == rank:
                candidate = [old if new == -1 else new for old, new in zip(shape, dimensions)]
                if all(
                    (type(new) is int and new > 0 or new == "B") and (old == new or old == 1)
                    for old, new in zip(shape, candidate)
                ):
                    result = candidate
        elif kind == "concatenation" and method == "cat" and inputs:
            axis = metadata.get("dimension")
            if type(axis) is int and -rank <= axis < rank:
                axis %= rank
                if all(
                    item.shape is not None
                    and len(item.shape) == rank
                    and all(v == shape[i] for i, v in enumerate(item.shape) if i != axis)
                    and type(item.shape[axis]) is int
                    for item in inputs
                ):
                    result = list(shape)
                    result[axis] = sum(item.shape[axis] for item in inputs)
    return TensorSpec(result, dtype, "static_transformation")
