"""Static tensor contracts with explicit validity, input constraints and unknowns."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch

from .ir import TensorSpec


@dataclass
class ShapeResult:
    spec: TensorSpec
    status: str
    rule: str
    reason: str = ""
    constraints: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {"rule": self.rule, "status": self.status, "reason": self.reason,
                "constraints": self.constraints, "output_shape": self.spec.shape,
                "output_dtype": self.spec.dtype, "runtime_verified": False}


def result(shape: list[Any] | None, dtype: str | None, rule: str, *, status: str = "proven",
           reason: str = "", constraints: list[str] | None = None) -> ShapeResult:
    if shape is not None and not known(shape):
        shape, status, reason = None, "unknown", reason or "Output contains an unresolved dimension"
    if status in {"proven", "conditional"} and dtype is None:
        status, reason = "unknown", reason or "Output dtype has not been established"
    return ShapeResult(TensorSpec(shape, dtype, "static_tensor"), status, rule, reason, constraints or [])


def known(shape: list[Any] | None) -> bool:
    return shape is not None and all(type(d) is int and d >= 0 or type(d) is str and d not in {"", "?", "unknown", "None"} for d in shape)


def equal(a: Any, b: Any) -> bool | None:
    if a == b:
        return True
    return False if type(a) is int and type(b) is int else None


def product(dimensions: list[Any]) -> Any:
    coefficient, symbols = 1, []
    for d in dimensions:
        if type(d) is int:
            coefficient *= d
        else:
            symbols.append(str(d))
    if not symbols or coefficient == 0:
        return coefficient
    return " * ".join(([str(coefficient)] if coefficient != 1 else []) + sorted(symbols))


def quotient(numerator: list[Any], denominator: list[Any]) -> tuple[Any | None, list[str]]:
    """Cancel only recorded factors, retaining nonzero requirements for symbolic divisors."""
    n, d, ns, ds = 1, 1, Counter(), Counter()
    for dims, symbols, numerator_side in ((numerator, ns, True), (denominator, ds, False)):
        for item in dims:
            if type(item) is int:
                if numerator_side:
                    n *= item
                else:
                    d *= item
            else:
                symbols[str(item)] += 1
    if d == 0 or n == 0 and ds:
        return None, []
    if ds - ns or n % d:
        return None, []
    constraints = [f"{symbol} != 0" for symbol in sorted(ds)]
    remaining = list((ns - ds).elements())
    return product([n // d, *remaining]), constraints


def broadcast(specs: list[TensorSpec], rule: str = "broadcast") -> ShapeResult:
    if not specs or any(not known(s.shape) for s in specs):
        return result(None, specs[0].dtype if specs else None, rule, status="unknown", reason="Input shape is unknown")
    shape: list[Any] = []
    for spec in specs:
        width = max(len(shape), len(spec.shape))
        left, right = [1] * (width - len(shape)) + shape, [1] * (width - len(spec.shape)) + spec.shape
        output = []
        for a, b in zip(left, right):
            if a == 1:
                output.append(b)
            elif b == 1 or a == b:
                output.append(a)
            else:
                return result(None, specs[0].dtype, rule, status="invalid" if equal(a, b) is False else "unknown",
                              reason=f"Broadcast dimensions disagree or are unproven: {a}, {b}")
        shape = output
    dtype = specs[0].dtype
    if any(s.dtype != dtype for s in specs):
        return result(shape, None, rule, status="unknown", reason="Mixed dtype promotion has not been established")
    return result(shape, dtype, rule)


def reshape(spec: TensorSpec, dimensions: list[Any]) -> ShapeResult:
    rule = "reshape_element_count"
    if not known(spec.shape) or not dimensions or dimensions.count(-1) > 1 or any(type(d) is int and d < -1 or type(d) not in (int, str) for d in dimensions):
        return result(None, spec.dtype, rule, status="unknown", reason="Reshape dimensions are missing or unsupported")
    dims = list(dimensions)
    if -1 in dims:
        inferred, conditions = quotient(spec.shape, [d for d in dims if d != -1])
        if inferred is None:
            return result(None, spec.dtype, rule, status="invalid" if all(type(d) is int for d in spec.shape + dims) else "unknown",
                          reason="Inferred dimension is ambiguous, nonintegral or unproven")
        dims[dims.index(-1)] = inferred
        return result(dims, spec.dtype, rule, status="conditional" if conditions else "proven", constraints=conditions)
    a, b = product(spec.shape), product(dims)
    if a == b:
        return result(dims, spec.dtype, rule)
    return result(None, spec.dtype, rule, status="invalid" if type(a) is int and type(b) is int else "unknown",
                  reason=f"Element counts differ or are unproven: {a}, {b}")


def transform(spec: TensorSpec, method: str, dimensions: list[Any]) -> ShapeResult:
    rule = f"tensor_{method}"
    if not known(spec.shape):
        return result(None, spec.dtype, rule, status="unknown", reason="Input rank or dimensions are unknown")
    shape, rank = list(spec.shape), len(spec.shape)
    if method in {"view", "reshape"}:
        return reshape(spec, dimensions)
    if method == "transpose" and len(dimensions) == 2 and all(type(d) is int and -rank <= d < rank for d in dimensions):
        a, b = (d % rank for d in dimensions)
        shape[a], shape[b] = shape[b], shape[a]
    elif method == "permute" and len(dimensions) == rank and all(type(d) is int and -rank <= d < rank for d in dimensions) and len({d % rank for d in dimensions}) == rank:
        shape = [shape[d % rank] for d in dimensions]
    elif method == "unsqueeze" and len(dimensions) == 1 and type(dimensions[0]) is int and -rank - 1 <= dimensions[0] <= rank:
        shape.insert(dimensions[0] % (rank + 1), 1)
    elif method == "squeeze" and (not dimensions or len(dimensions) == 1 and type(dimensions[0]) is int and -rank <= dimensions[0] < rank):
        axes = range(rank) if not dimensions else [dimensions[0] % rank]
        if any(type(shape[i]) is not int for i in axes):
            return result(None, spec.dtype, rule, status="unknown", reason="Squeeze rank depends on an unknown unit dimension")
        shape = [d for i, d in enumerate(shape) if i not in axes or d != 1]
    elif method == "flatten" and len(dimensions) <= 2:
        start, end = (dimensions + [-1])[:2] if dimensions else (0, -1)
        if not all(type(d) is int and -rank <= d < rank for d in (start, end)) or start % rank > end % rank:
            return result(None, spec.dtype, rule, status="invalid", reason="Flatten axes are invalid")
        start, end = start % rank, end % rank
        shape = shape[:start] + [product(shape[start:end + 1])] + shape[end + 1:]
    elif method == "expand" and len(dimensions) >= rank:
        old = [1] * (len(dimensions) - rank) + shape
        candidate = []
        for i, (a, b) in enumerate(zip(old, dimensions)):
            if b == -1:
                if i < len(dimensions) - rank:
                    return result(None, spec.dtype, rule, status="invalid", reason="Leading expand dimensions cannot be inferred")
                b = a
            if type(b) is int and b < 0:
                return result(None, spec.dtype, rule, status="invalid", reason="Negative expand extent")
            if a != 1 and equal(a, b) is not True:
                return result(None, spec.dtype, rule, status="invalid" if equal(a, b) is False else "unknown", reason="Non-unit expand dimension is incompatible")
            candidate.append(b)
        shape = candidate
    elif method in {"contiguous", "to", "type_as", "float", "half"}:
        pass
    else:
        return result(None, spec.dtype, rule, status="invalid" if all(type(d) is int for d in dimensions) else "unknown", reason="Axes or arguments are invalid or unsupported")
    return result(shape, spec.dtype, rule)


def selection(spec: TensorSpec, indices: list[dict[str, Any]]) -> ShapeResult:
    rule = "tensor_indexing"
    if not known(spec.shape):
        return result(None, spec.dtype, rule, status="unknown", reason="Indexed input shape is unknown")
    shape, axis, conditions = [], 0, []
    for index in indices:
        kind = index.get("kind")
        if kind == "new_axis":
            shape.append(1)
            continue
        if axis >= len(spec.shape):
            return result(None, spec.dtype, rule, status="invalid", reason="Too many indices")
        extent = spec.shape[axis]
        axis += 1
        if kind == "integer":
            value = index["value"]
            if type(extent) is int and not -extent <= value < extent:
                return result(None, spec.dtype, rule, status="invalid", reason="Integer index exceeds axis bounds")
            if type(extent) is not int:
                conditions.append(f"{-value if value < 0 else value + 1} <= {extent}")
        elif kind == "slice":
            start, stop, step = index.get("start"), index.get("stop"), index.get("step")
            if step == 0:
                return result(None, spec.dtype, rule, status="invalid", reason="Slice step is zero")
            if all(type(v) in (int, type(None)) for v in (start, stop, step)) and type(extent) is int:
                shape.append(len(range(*slice(start, stop, step).indices(extent))))
            elif start in (None, 0) and stop is None and step in (None, 1):
                shape.append(extent)
            elif start in (None, 0) and type(stop) is str and step in (None, 1) and type(extent) is int:
                shape.append(stop)
                conditions.extend([f"0 <= {stop}", f"{stop} <= {extent}"])
            else:
                return result(None, spec.dtype, rule, status="unknown", reason="Symbolic slice length has not been established")
        else:
            return result(None, spec.dtype, rule, status="unknown", reason="Advanced or dynamic indexing is unsupported")
    return result(shape + spec.shape[axis:], spec.dtype, rule, status="conditional" if conditions else "proven", constraints=conditions)


def matmul(left: TensorSpec, right: TensorSpec, *, bmm: bool = False) -> ShapeResult:
    rule = "bmm_contract" if bmm else "matmul_contract"
    if left.dtype == "BOOL" or right.dtype == "BOOL":
        return result(None, None, rule, status="unknown", reason="Boolean matmul dtype support has not been established")
    if not known(left.shape) or not known(right.shape) or not left.shape or not right.shape:
        return result(None, left.dtype, rule, status="unknown", reason="Matmul requires established positive ranks")
    a, b = list(left.shape), list(right.shape)
    if bmm and (len(a) != 3 or len(b) != 3 or equal(a[0], b[0]) is not True):
        return result(None, left.dtype, rule, status="invalid" if len(a) != 3 or len(b) != 3 or equal(a[0], b[0]) is False else "unknown", reason="bmm requires equal batch extents and rank three")
    vector_a, vector_b = len(a) == 1, len(b) == 1
    if vector_a:
        a.insert(0, 1)
    if vector_b:
        b.append(1)
    contraction = equal(a[-1], b[-2])
    if contraction is not True:
        return result(None, left.dtype, rule, status="invalid" if contraction is False else "unknown", reason="Contraction dimensions differ or are unproven")
    batch = broadcast([TensorSpec(a[:-2], left.dtype, "batch"), TensorSpec(b[:-2], right.dtype, "batch")], rule)
    if batch.status not in {"proven", "conditional"}:
        return batch
    return result(batch.spec.shape + ([] if vector_a else [a[-2]]) + ([] if vector_b else [b[-1]]), batch.spec.dtype, rule)


def concatenate(specs: list[TensorSpec], axis: Any, *, stack: bool = False) -> ShapeResult:
    rule = "stack_dimensions" if stack else "concatenate_dimensions"
    if not specs or any(not known(spec.shape) for spec in specs) or type(axis) is not int:
        return result(None, None, rule, status="unknown", reason="Concatenation shapes or axis are unproven")
    shape, dtype = list(specs[0].shape), specs[0].dtype
    rank = len(shape) + int(stack)
    if rank == 0 or not -rank <= axis < rank:
        return result(None, dtype, rule, status="invalid", reason="Concatenation axis exceeds rank")
    axis %= rank
    for spec in specs:
        if len(spec.shape) != len(shape) or any(equal(a, b) is not True for i, (a, b) in enumerate(zip(shape, spec.shape)) if stack or i != axis):
            return result(None, dtype, rule, status="unknown", reason="Nonconcatenated dimensions differ or are unproven")
        if spec.dtype != dtype:
            return result(None, None, rule, status="unknown", reason="Concatenation dtype promotion is unproven")
    if stack:
        shape.insert(axis, len(specs))
    else:
        extents = [spec.shape[axis] for spec in specs]
        shape[axis] = sum(extents) if all(type(d) is int for d in extents) else " + ".join(str(d) for d in extents)
    return result(shape, dtype, rule)


def module_contract(module: torch.nn.Module, incoming: TensorSpec | None) -> ShapeResult | None:
    """Use exact native classes and real weight layouts, never semantic class labels."""
    from transformers.pytorch_utils import Conv1D
    supported = {torch.nn.Embedding, torch.nn.Linear, Conv1D, torch.nn.Conv1d,
                 torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm, torch.nn.Dropout,
                 torch.nn.Identity, torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU,
                 torch.nn.Tanh, torch.nn.Sigmoid}
    if type(module) not in supported:
        return None
    if "forward" in vars(module) or module._forward_hooks or module._forward_pre_hooks:
        return result(None, None, "native_module_override", status="unknown", reason="Forward override or hooks invalidate the native shape contract")
    if incoming is None or not known(incoming.shape):
        return result(None, incoming.dtype if incoming else None, "native_module", status="unknown", reason="Module input shape is not established")
    shape, dtype = list(incoming.shape), incoming.dtype
    module_type = type(module)
    if module_type is torch.nn.Embedding:
        if dtype not in {"I32", "I64"}:
            return result(None, dtype, "embedding_indices", status="invalid" if dtype is not None else "unknown", reason="Embedding indices require I32 or I64")
        from .source_eval import DTYPES
        return result(shape + [module.weight.shape[1]], DTYPES.get(module.weight.dtype), "native_embedding",
                      constraints=[f"0 <= input_values < {module.num_embeddings}"], status="conditional")
    if module_type in {torch.nn.Linear, Conv1D}:
        features = module.weight.shape[1] if module_type is torch.nn.Linear else module.weight.shape[0]
        width = module.weight.shape[0] if module_type is torch.nn.Linear else module.weight.shape[1]
        same = equal(shape[-1], features) if shape else False
        if same is not True:
            return result(None, dtype, "native_projection", status="invalid" if same is False else "unknown", reason=f"Input feature dimension must equal native weight feature dimension {features}")
        from .source_eval import DTYPES
        native_dtype = DTYPES.get(module.weight.dtype)
        if dtype != native_dtype:
            return result(None, None, "native_projection", status="invalid" if dtype and native_dtype else "unknown", reason="Projection input/weight dtype compatibility is not established; autocast is outside this profile")
        return result(shape[:-1] + [width], native_dtype, "native_projection")
    if module_type in {torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d}:
        from .source_eval import DTYPES
        if dtype != DTYPES.get(module.weight.dtype):
            return result(None, None, "native_convolution", status="invalid" if dtype else "unknown", reason="Convolution input/weight dtype compatibility is not established")
        dimensions = len(module.kernel_size)
        if len(shape) not in {dimensions + 1, dimensions + 2}:
            return result(None, dtype, "native_convolution", status="invalid", reason="Convolution rank is incompatible")
        channel_axis = len(shape) - dimensions - 1
        same = equal(shape[channel_axis], module.in_channels)
        if same is not True:
            return result(None, dtype, "native_convolution", status="invalid" if same is False else "unknown", reason="Convolution channel count is incompatible or unproven")
        if isinstance(module.padding, str):
            return result(None, dtype, "native_convolution", status="unknown", reason="String convolution padding is not yet supported")
        spatial = []
        for size, pad, dilation, kernel, stride in zip(shape[-dimensions:], module.padding, module.dilation, module.kernel_size, module.stride):
            offset = 2 * pad - dilation * (kernel - 1) - 1
            if type(size) is int:
                value = (size + offset) // stride + 1
                if value <= 0:
                    return result(None, dtype, "native_convolution", status="invalid", reason="Convolution output extent is not positive")
                spatial.append(value)
            else:
                return result(None, dtype, "native_convolution", status="unknown", reason="Symbolic convolution extent requires unresolved floor/bound constraints")
        return result(shape[:channel_axis] + [module.out_channels] + spatial, dtype, "native_convolution")
    if module_type is torch.nn.LayerNorm:
        if dtype not in {"F16", "BF16", "F32", "F64"}:
            return result(None, None, "layer_norm_dtype", status="unknown", reason="LayerNorm dtype compatibility is not established")
        expected = list(module.normalized_shape)
        dimensions = [equal(a, b) for a, b in zip(shape[-len(expected):], expected)]
        if len(shape) < len(expected) or any(item is not True for item in dimensions):
            return result(None, dtype, "layer_norm_tail", status="invalid" if len(shape) < len(expected) or False in dimensions else "unknown", reason="Normalized trailing dimensions differ or are unproven")
        return result(shape, dtype, "layer_norm_tail")
    if module_type in {torch.nn.Dropout, torch.nn.Identity, torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Tanh, torch.nn.Sigmoid}:
        if module_type in {torch.nn.GELU, torch.nn.SiLU, torch.nn.Tanh, torch.nn.Sigmoid} and dtype not in {"F16", "BF16", "F32", "F64"}:
            return result(None, None, "native_activation_dtype", status="unknown", reason="Activation dtype compatibility is not established")
        return result(shape, dtype, "native_elementwise_shape")
    return None
