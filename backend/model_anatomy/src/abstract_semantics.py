"""Shared abstract values and bounded Python semantics; never executes tensor operators."""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from typing import Any


class SourceBoundary(Exception):
    """A dependency cannot be established within the supported static semantics."""


class MissingAttribute(SourceBoundary):
    """Ordinary static attribute absence, distinct from dynamic lookup."""


@dataclass(frozen=True)
class Symbol:
    expression: str
    scalar_type: str | None = None

    def __post_init__(self) -> None:
        if len(self.expression) > 2048:
            raise SourceBoundary("Symbol exceeds its expression budget")


@dataclass(frozen=True)
class Tensor:
    shape: tuple[Any, ...]
    dtype: str
    expression: str

    def __post_init__(self) -> None:
        if len(self.shape) > 16 or len(self.expression) > 16384:
            raise SourceBoundary("Symbolic tensor exceeds its rank or expression budget")
        if self.dtype not in {"F16", "BF16", "F32", "F64", "BOOL", "I32", "I64"}:
            raise SourceBoundary("Symbolic tensor dtype is unsupported")
        if any(not isinstance(d, Symbol) and (type(d) is not int or d < 0) for d in self.shape):
            raise SourceBoundary("Tensor dimensions are not nonnegative integers or symbolic extents")

    def spec(self) -> dict[str, Any]:
        return {"shape": [d.expression if isinstance(d, Symbol) else d for d in self.shape],
                "dtype": self.dtype, "expression": self.expression}


@dataclass
class Closure:
    tree: ast.FunctionDef
    env: dict[str, Any]
    globals: dict[str, Any]
    source: dict[str, Any]
    defaults: dict[str, Any] = field(default_factory=dict)


def scalar_binary(left: Any, right: Any, op: ast.AST) -> Any:
    """Apply operator-specific budgets to proven scalar literals only."""
    if type(left) not in (int, float, bool) or type(right) not in (int, float, bool):
        raise SourceBoundary("Arithmetic operands are not scalar literals")
    for value in (left, right):
        if type(value) is int and value.bit_length() > 64 or type(value) is float and not math.isfinite(value):
            raise SourceBoundary("Scalar operand exceeds its numeric budget")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                  ast.Pow: operator.pow, ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
    if type(op) not in operations:
        raise SourceBoundary("Unsupported scalar arithmetic operator")
    if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor)) and (type(left) is not bool or type(right) is not bool):
        raise SourceBoundary("Only proven Python booleans support static bitwise operations")
    if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
        raise SourceBoundary("Static division by zero is invalid")
    if isinstance(op, ast.Pow) and (abs(right) > 64 or type(left) is int and type(right) is int and right > 0 and left.bit_length() * right > 128):
        raise SourceBoundary("Exponentiation exceeds its bounded result budget")
    try:
        result = operations[type(op)](left, right)
    except (OverflowError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise SourceBoundary("Scalar arithmetic is invalid") from exc
    if type(result) not in (int, float, bool) or type(result) is int and result.bit_length() > 64 or type(result) is float and not math.isfinite(result):
        raise SourceBoundary("Scalar arithmetic exceeds its result budget")
    return result


def bind_arguments(tree: ast.FunctionDef, args: list[Any], kwargs: dict[str, Any], defaults: dict[str, Any],
                   *, skip_self: bool = False) -> dict[str, Any]:
    """Bind AST arguments with actual Python positional-only/default/keyword rules."""
    params = tree.args
    positional = params.posonlyargs + params.args
    if skip_self and positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    if len(args) > len(positional) and params.vararg is None:
        raise SourceBoundary("Too many positional helper arguments")
    bound = {item.arg: value for item, value in zip(positional, args)}
    remaining = dict(kwargs)
    positional_only = {item.arg for item in params.posonlyargs}
    allowed = {item.arg for item in positional + params.kwonlyargs} - positional_only
    for name, value in kwargs.items():
        if name in allowed:
            if name in bound:
                raise SourceBoundary("Helper argument supplied more than once")
            bound[name] = value
            remaining.pop(name)
    for item in positional + params.kwonlyargs:
        if item.arg not in bound:
            if item.arg not in defaults:
                raise SourceBoundary(f"Required helper argument is unresolved: {item.arg}")
            bound[item.arg] = defaults[item.arg]
    if params.vararg:
        bound[params.vararg.arg] = tuple(args[len(positional):])
    if remaining and params.kwarg is None:
        raise SourceBoundary("Unexpected helper keyword arguments")
    if params.kwarg:
        bound[params.kwarg.arg] = remaining
    return bound


def metadata_primitive(function: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Shared identity-based metadata primitives over known abstract structures."""
    if kwargs:
        return NotImplemented
    if function is len and len(args) == 1 and type(args[0]) in (tuple, list, dict, str):
        return len(args[0])
    if function is type and len(args) == 1:
        value = args[0]
        if type(value) in (type(None), bool, int, float, str, list, tuple, dict):
            return type(value)
        if isinstance(value, Tensor):
            return NotImplemented
    if function is math.sqrt and len(args) == 1 and type(args[0]) in (int, float) and math.isfinite(args[0]) and args[0] >= 0:
        return math.sqrt(args[0])
    return NotImplemented
