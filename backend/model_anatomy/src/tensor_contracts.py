"""Conservative tensor shape and dtype contracts for a small operator subset.

This module reasons over IR metadata only. It never imports or executes Torch.
An unsupported or under-specified call returns ``None`` so callers can retain
the operation without claiming an output contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, isfinite
from typing import Any

from .ir import TensorSpec

_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
_KNOWN_DTYPES = _FLOAT_DTYPES | {"BOOL", "U8", "I8", "I16", "I32", "I64"}
_FLOAT_MAX = {
    "F16": 65504.0,
    "BF16": 3.3895313892515355e38,
    "F32": 3.4028234663852886e38,
    "F64": 1.7976931348623157e308,
}


def tensor_outputs(
    method: str, inputs: dict[str, TensorSpec], parameters: dict[str, Any]
) -> dict[str, TensorSpec] | None:
    """Return output ports when the named operator contract is fully provable."""
    if not isinstance(method, str) or not isinstance(inputs, dict) or not isinstance(parameters, dict):
        return None

    if method == "isinf":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_no_parameters(parameters):
            return None
        value = inputs["input"]
        if not _valid_spec(value) or value.dtype not in _FLOAT_DTYPES:
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), "BOOL", "isinf")}

    if method == "chunk":
        return _chunk_outputs(inputs, parameters)

    if method == "clone":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_no_parameters(parameters):
            return None
        value = inputs["input"]
        if not _valid_spec(value):
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), value.dtype, "clone")}

    if method == "triu":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"diagonal"}):
            return None
        value = inputs["input"]
        if (
            not _valid_spec(value)
            or value.shape is None
            or len(value.shape) < 2
            or value.dtype not in _FLOAT_DTYPES
            or type(parameters["diagonal"]) is not int
        ):
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), value.dtype, "triu")}

    if method in {"gt", "eq", "float_bool_mul"}:
        if not _has_exact_inputs(inputs, {"left", "right"}) or not _has_no_parameters(parameters):
            return None
        left, right = inputs["left"], inputs["right"]
        if not all(_valid_spec(item) and item.shape is not None for item in (left, right)):
            return None
        shape = _broadcast_shapes(left.shape, right.shape)
        if shape is _UNKNOWN:
            return None
        if (
            method == "gt" and left.dtype == right.dtype == "I64"
            or method == "eq" and left.dtype == right.dtype and left.dtype in _KNOWN_DTYPES
        ):
            dtype = "BOOL"
        elif method == "float_bool_mul" and left.dtype in _FLOAT_DTYPES and right.dtype == "BOOL":
            dtype = left.dtype
        else:
            return None
        return {"output": TensorSpec(shape, dtype, method)}

    if method == "float_int_add":
        if not _has_exact_inputs(inputs, {"left", "right"}) or not _has_no_parameters(parameters):
            return None
        left, right = inputs["left"], inputs["right"]
        if (
            not _valid_spec(left)
            or not _valid_spec(right)
            or left.shape is None
            or right.shape is None
            or left.dtype not in _FLOAT_DTYPES
            or right.dtype != "I64"
        ):
            return None
        shape = _broadcast_shapes(left.shape, right.shape)
        if shape is _UNKNOWN:
            return None
        return {"output": TensorSpec(shape, left.dtype, "float_int_add")}

    if method == "scalar_left_sub":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"scalar"}):
            return None
        value = inputs["input"]
        scalar = parameters["scalar"]
        if (
            not _valid_spec(value)
            or value.shape is None
            or value.dtype not in _FLOAT_DTYPES
            or type(scalar) not in {int, float}
            or not _representable_float(scalar, value.dtype)
        ):
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), value.dtype, "scalar_left_sub")}

    if method == "eq_scalar":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"scalar"}):
            return None
        value = inputs["input"]
        scalar = parameters["scalar"]
        if (
            not _valid_spec(value)
            or value.shape is None
            or value.dtype not in _FLOAT_DTYPES
            or type(scalar) not in {int, float}
            or not _representable_float(scalar, value.dtype)
        ):
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), "BOOL", "eq_scalar")}

    if method == "expand":
        if (
            not _has_exact_inputs(inputs, {"input"})
            or not _has_exact_keys(parameters, {"dimensions", "positive_symbols"})
        ):
            return None
        value = inputs["input"]
        dimensions = parameters["dimensions"]
        positive_symbols = parameters["positive_symbols"]
        if (
            not _valid_spec(value)
            or value.shape is None
            or not _valid_expand_dimensions(dimensions)
            or not _valid_positive_symbols(positive_symbols)
            or not dimensions
            or any(isinstance(axis, str) and axis not in positive_symbols for axis in dimensions)
        ):
            return None
        shape = _expanded_shape(value.shape, dimensions)
        if shape is None:
            return None
        return {"output": TensorSpec(shape, value.dtype, "expand")}

    if method == "slice":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"indices"}):
            return None
        value = inputs["input"]
        if not _valid_spec(value) or value.shape is None:
            return None
        shape = _slice_shape(value.shape, parameters["indices"])
        if shape is None:
            return None
        return {"output": TensorSpec(shape, value.dtype, "slice")}

    if method == "masked_fill":
        if not _has_exact_inputs(inputs, {"input", "mask"}) or not _has_exact_keys(parameters, {"value"}):
            return None
        value, mask = inputs["input"], inputs["mask"]
        fill = parameters["value"]
        if (
            not _valid_spec(value)
            or not _valid_spec(mask)
            or value.shape is None
            or value.dtype not in _FLOAT_DTYPES
            or mask.dtype != "BOOL"
            or type(fill) not in {int, float}
            or not _representable_float(fill, value.dtype)
        ):
            return None
        broadcast = _broadcast_shapes(value.shape, mask.shape)
        if broadcast is _UNKNOWN or broadcast != value.shape:
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), value.dtype, "masked_fill")}

    if method == "clamp":
        if not _has_exact_inputs(inputs, {"input"}) or not _allowed_keys(parameters, {"min", "max"}):
            return None
        value = inputs["input"]
        if not _valid_spec(value) or value.dtype not in _FLOAT_DTYPES:
            return None
        lower, upper = parameters.get("min"), parameters.get("max")
        if lower is None and upper is None:
            return None
        if not _scalar_bound(lower) or not _scalar_bound(upper):
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), value.dtype, "clamp")}

    if method == "where":
        if not _has_exact_inputs(inputs, {"condition", "input", "other"}) or not _has_no_parameters(parameters):
            return None
        condition, left, right = inputs["condition"], inputs["input"], inputs["other"]
        if (
            not all(_valid_spec(item) for item in (condition, left, right))
            or condition.dtype != "BOOL"
            or left.dtype not in _KNOWN_DTYPES
            or right.dtype != left.dtype
        ):
            return None
        shape = _broadcast_shapes(condition.shape, left.shape, right.shape)
        if shape is _UNKNOWN:
            return None
        return {"output": TensorSpec(shape, left.dtype, "where")}

    if method == "any":
        if not _has_exact_inputs(inputs, {"input"}) or not _allowed_keys(parameters, {"dim", "keepdim"}):
            return None
        value = inputs["input"]
        if not _valid_spec(value) or value.dtype not in _KNOWN_DTYPES or value.shape is None:
            return None
        keepdim = parameters.get("keepdim", False)
        if type(keepdim) is not bool:
            return None
        axes = _reduction_axes(parameters.get("dim"), len(value.shape))
        if axes is None:
            return None
        if keepdim:
            shape = list(value.shape)
            for axis in axes:
                shape[axis] = 1
        else:
            shape = [size for axis, size in enumerate(value.shape) if axis not in axes]
        return {"output": TensorSpec(shape, "BOOL", "any")}

    if method in {"reshape", "view"}:
        if (
            not _has_exact_inputs(inputs, {"input"})
            or "dimensions" not in parameters
            or not _allowed_keys(parameters, {"dimensions", "positive_symbols"})
        ):
            return None
        value = inputs["input"]
        dimensions = parameters["dimensions"]
        positive_symbols = parameters.get("positive_symbols", [])
        if (
            not _valid_spec(value)
            or value.shape is None
            or not _valid_dimensions(dimensions)
            or not _valid_positive_symbols(positive_symbols)
        ):
            return None
        shape = _reshape_shape(value.shape, dimensions, set(positive_symbols))
        if shape is None:
            return None
        return {"output": TensorSpec(shape, value.dtype, method)}

    if method == "transpose":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"dimensions"}):
            return None
        value = inputs["input"]
        dimensions = parameters["dimensions"]
        if (
            not _valid_spec(value)
            or value.shape is None
            or not isinstance(dimensions, list)
            or len(dimensions) != 2
            or any(type(axis) is not int for axis in dimensions)
        ):
            return None
        rank = len(value.shape)
        if any(not -rank <= axis < rank for axis in dimensions):
            return None
        left, right = (axis % rank for axis in dimensions) if rank else (0, 0)
        shape = list(value.shape)
        shape[left], shape[right] = shape[right], shape[left]
        return {"output": TensorSpec(shape, value.dtype, "transpose")}

    if method == "unsqueeze":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"dim"}):
            return None
        value = inputs["input"]
        dim = parameters["dim"]
        if not _valid_spec(value) or value.shape is None or type(dim) is not int:
            return None
        rank = len(value.shape)
        if not -(rank + 1) <= dim < rank + 1:
            return None
        shape = list(value.shape)
        shape.insert(dim % (rank + 1), 1)
        return {"output": TensorSpec(shape, value.dtype, "unsqueeze")}

    if method == "squeeze":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"dim"}):
            return None
        value = inputs["input"]
        dim = parameters["dim"]
        if not _valid_spec(value) or value.shape is None or type(dim) is not int:
            return None
        rank = len(value.shape)
        if not -rank <= dim < rank:
            return None
        axis = dim % rank if rank else dim
        if rank and value.shape[axis] == 1:
            shape = value.shape[:axis] + value.shape[axis + 1 :]
        elif rank and type(value.shape[axis]) is int and value.shape[axis] != 1:
            shape = list(value.shape)
        else:
            return None
        return {"output": TensorSpec(shape, value.dtype, "squeeze")}

    if method == "flatten":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"start_dim", "end_dim"}):
            return None
        value = inputs["input"]
        start, end = parameters["start_dim"], parameters["end_dim"]
        if (
            not _valid_spec(value)
            or value.shape is None
            or type(start) is not int
            or type(end) is not int
            or not value.shape
        ):
            return None
        rank = len(value.shape)
        if not -rank <= start < rank or not -rank <= end < rank:
            return None
        start, end = start % rank, end % rank
        if start > end:
            return None
        flattened = _flatten_dimension(value.shape[start : end + 1])
        if flattened is None:
            return None
        shape = value.shape[:start] + [flattened] + value.shape[end + 1 :]
        return {"output": TensorSpec(shape, value.dtype, "flatten")}

    if method == "cast":
        if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"dtype"}):
            return None
        value = inputs["input"]
        dtype = parameters["dtype"]
        if not _valid_spec(value) or not isinstance(dtype, str) or dtype not in _KNOWN_DTYPES:
            return None
        return {"output": TensorSpec(_copy_shape(value.shape), dtype, "cast")}

    return None


_UNKNOWN = object()


def _chunk_outputs(
    inputs: dict[str, TensorSpec], parameters: dict[str, Any]
) -> dict[str, TensorSpec] | None:
    if not _has_exact_inputs(inputs, {"input"}) or not _has_exact_keys(parameters, {"chunks", "dim"}):
        return None
    value = inputs["input"]
    chunks, dim = parameters["chunks"], parameters["dim"]
    if not _valid_spec(value) or value.dtype not in _KNOWN_DTYPES or value.shape is None:
        return None
    if type(chunks) is not int or chunks <= 0 or type(dim) is not int or not value.shape:
        return None
    rank = len(value.shape)
    if not -rank <= dim < rank:
        return None
    axis = dim % rank
    extent = value.shape[axis]
    if type(extent) is not int or extent < 0:
        return None

    # torch.chunk uses ceil(extent / chunks) as its split size. It can return
    # fewer than ``chunks`` tensors when extent is not evenly divisible.
    split_size = ceil(extent / chunks) if extent else 0
    count = max(1, ceil(extent / split_size)) if split_size else chunks
    outputs: dict[str, TensorSpec] = {}
    for index in range(count):
        start = index * split_size
        end = min(start + split_size, extent)
        shape = list(value.shape)
        shape[axis] = end - start
        outputs[str(index)] = TensorSpec(shape, value.dtype, "chunk")
    return outputs


def _reduction_axes(dim: Any, rank: int) -> set[int] | None:
    if dim is None:
        return set(range(rank))
    if type(dim) is int:
        dims = [dim]
    elif isinstance(dim, (tuple, list)) and all(type(axis) is int for axis in dim):
        dims = list(dim)
    else:
        return None
    axes: set[int] = set()
    for axis in dims:
        if not -rank <= axis < rank:
            return None
        normalized = axis % rank if rank else axis
        if normalized in axes:
            return None
        axes.add(normalized)
    return axes


def _broadcast_shapes(*shapes: list[int | str] | None) -> list[int | str] | object:
    if any(shape is None for shape in shapes):
        return _UNKNOWN
    rank = max((len(shape) for shape in shapes if shape is not None), default=0)
    result: list[int | str] = []
    for offset in range(1, rank + 1):
        dimensions = [shape[-offset] if len(shape) >= offset else 1 for shape in shapes]  # type: ignore[arg-type]
        merged: int | str = 1
        for dimension in dimensions:
            if dimension == 1:
                continue
            if merged == 1 or merged == dimension:
                merged = dimension
            else:
                # A symbol could equal a concrete dimension at runtime, but
                # without a constraint that compatibility is not established.
                return _UNKNOWN
        result.append(merged)
    result.reverse()
    return result


def _expanded_shape(
    input_shape: list[int | str], dimensions: list[int | str]
) -> list[int | str] | None:
    if len(dimensions) < len(input_shape):
        return None
    padded: list[int | str | None] = [None] * (len(dimensions) - len(input_shape)) + list(input_shape)
    output: list[int | str] = []
    for source, target in zip(padded, dimensions):
        if source is None:
            if target == -1:
                return None
            output.append(target)
        elif target == -1:
            output.append(source)
        elif type(source) is int:
            if source == target or source == 1:
                output.append(target)
            else:
                return None
        elif source == target or source == 1:
            output.append(target)
        else:
            return None
    return output


def _slice_shape(shape: list[int | str], indices: Any) -> list[int | str] | None:
    if not isinstance(indices, list) or not indices:
        return None
    result: list[int | str] = []
    axis = 0
    for index in indices:
        if index is None:
            result.append(1)
            continue
        if (
            not isinstance(index, dict)
            or set(index) != {"start", "stop", "step"}
            or axis >= len(shape)
        ):
            return None
        start, stop, step = index["start"], index["stop"], index["step"]
        if (start is not None and (type(start) is not int or start != 0)) or (
            step is not None and (type(step) is not int or step != 1)
        ):
            return None
        dimension = shape[axis]
        if stop is None or stop == dimension:
            result.append(dimension)
        elif type(dimension) is int and type(stop) is int and stop >= 0:
            result.append(len(range(*slice(start, stop, step).indices(dimension))))
        else:
            return None
        axis += 1
    if axis != len(shape):
        return None
    return result


def _valid_spec(value: Any) -> bool:
    if not isinstance(value, TensorSpec) or value.dtype not in _KNOWN_DTYPES:
        return False
    if value.shape is None:
        return True
    return all(type(axis) is int and axis >= 0 or isinstance(axis, str) and bool(axis) for axis in value.shape)


def _copy_shape(shape: list[int | str] | None) -> list[int | str] | None:
    return list(shape) if shape is not None else None


def _valid_dimensions(dimensions: Any) -> bool:
    return (
        isinstance(dimensions, list)
        and all(
            type(axis) is int and axis >= -1 or isinstance(axis, str) and bool(axis)
            for axis in dimensions
        )
        and sum(axis == -1 for axis in dimensions) <= 1
    )


def _valid_expand_dimensions(dimensions: Any) -> bool:
    return isinstance(dimensions, list) and all(
        type(axis) is int and axis >= -1 or isinstance(axis, str) and axis.isidentifier()
        for axis in dimensions
    )


def _valid_positive_symbols(symbols: Any) -> bool:
    return isinstance(symbols, list) and all(
        isinstance(symbol, str) and symbol.isidentifier() for symbol in symbols
    ) and len(set(symbols)) == len(symbols)


def _reshape_shape(
    input_shape: list[int | str], dimensions: list[int | str], positive_symbols: set[str]
) -> list[int | str] | None:
    if -1 not in dimensions:
        if dimensions == input_shape:
            return list(dimensions)
        if all(type(axis) is int for axis in input_shape + dimensions):
            return list(dimensions) if _product(input_shape) == _product(dimensions) else None
        return None

    target = list(dimensions)
    missing = target.index(-1)
    known_target = target[:missing] + target[missing + 1 :]
    if all(type(axis) is int for axis in input_shape + known_target):
        denominator = _product(known_target)
        numerator = _product(input_shape)
        if denominator == 0 or numerator % denominator:
            return None
        target[missing] = numerator // denominator
        return target

    source = list(input_shape)
    target_factors = list(known_target)
    for symbol in {axis for axis in source if isinstance(axis, str)}:
        source_count = source.count(symbol)
        target_count = target_factors.count(symbol)
        if target_count and symbol not in positive_symbols:
            continue
        for _ in range(min(source_count, target_count)):
            source.remove(symbol)
            target_factors.remove(symbol)

    symbols = [axis for axis in source if isinstance(axis, str)]
    if (
        len(symbols) > 1
        or not all(type(axis) is int or isinstance(axis, str) for axis in source)
        or not all(type(axis) is int for axis in target_factors)
    ):
        return None
    input_fixed = _product([axis for axis in source if type(axis) is int])
    target_fixed = _product([axis for axis in target_factors if type(axis) is int])
    if target_fixed == 0:
        return None
    if symbols:
        if input_fixed != target_fixed:
            return None
        target[missing] = symbols[0]
        return target
    if input_fixed % target_fixed:
        return None
    target[missing] = input_fixed // target_fixed
    return target


def _flatten_dimension(shape: list[int | str]) -> int | str | None:
    if all(type(axis) is int for axis in shape):
        return _product(shape)
    symbols = [axis for axis in shape if isinstance(axis, str)]
    fixed = [axis for axis in shape if type(axis) is int]
    if len(symbols) == 1 and _product(fixed) == 1:
        return symbols[0]
    return None


def _product(values: list[int | str]) -> int:
    product = 1
    for value in values:
        product *= value  # callers establish that these are integers
    return product


def _scalar_bound(value: Any) -> bool:
    return value is None or (type(value) in {int, float} and (type(value) is int or isfinite(value)))


def _representable_float(value: float, dtype: str) -> bool:
    try:
        return isfinite(value) and abs(value) <= _FLOAT_MAX[dtype]
    except (OverflowError, TypeError):
        return False


def _has_exact_inputs(inputs: Mapping[str, Any], expected: set[str]) -> bool:
    return set(inputs) == expected


def _has_no_parameters(parameters: Mapping[str, Any]) -> bool:
    return not parameters


def _allowed_keys(parameters: Mapping[str, Any], allowed: set[str]) -> bool:
    return set(parameters) <= allowed


def _has_exact_keys(parameters: Mapping[str, Any], expected: set[str]) -> bool:
    return set(parameters) == expected
