"""Identity-checked facts about native PyTorch tensor factories."""

from __future__ import annotations

import ast
import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

_FACTORIES = {
    name: getattr(torch, name)
    for name in ("ones", "zeros", "empty", "full", "ones_like", "zeros_like", "empty_like", "full_like")
}
_METHOD_FACTORIES = {"new_ones", "new_zeros", "new_empty", "new_full"}
_NATIVE_METHODS = {name: inspect.getattr_static(torch.Tensor, name) for name in _METHOD_FACTORIES}
_CANDIDATES = frozenset(_FACTORIES) | _METHOD_FACTORIES
_LIKE_FACTORIES = frozenset({"ones_like", "zeros_like", "empty_like", "full_like"})


@dataclass(frozen=True)
class TensorFactoryFact:
    name: str
    native: bool
    rank: int | None = None
    reason: str | None = None

    @property
    def assumption(self) -> dict[str, Any]:
        identity = f"torch.Tensor.{self.name}" if self.name in _METHOD_FACTORIES else f"torch.{self.name}"
        return {
            "predicate": f"torch.{self.name}",
            "rule_sha256": hashlib.sha256(f"native-torch-factory:{identity}".encode()).hexdigest(),
            "value": "tensor",
            "basis": f"Callable identity matched {identity}; {self.rank_basis}",
        }

    @property
    def rank_basis(self) -> str:
        return f"result rank is {self.rank}" if self.rank is not None else "result rank was not established"


def _callable_for(node: ast.expr, source_globals: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        return source_globals.get(node.id)
    chain: list[str] = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    value = source_globals.get(node.id)
    for part in reversed(chain):
        if isinstance(value, ModuleType):
            value = vars(value).get(part)
        elif value is torch.Tensor:
            value = inspect.getattr_static(value, part, None)
        else:
            return None
    return value


def _invalid_dimension(node: ast.expr) -> bool:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False
    return type(value) is not int or value < 0


def _rank_of_size(node: ast.expr | None, resolve_rank: Callable[[ast.expr], int | None]) -> int | None:
    if node is None:
        return None
    if isinstance(node, (ast.Tuple, ast.List)):
        if any(_invalid_dimension(item) for item in node.elts):
            return None
        return len(node.elts)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return 1 if node.value >= 0 else None
    return resolve_rank(node)


def _known_invalid_size(node: ast.expr | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return type(node.value) is not int or node.value < 0
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_invalid_dimension(item) for item in node.elts)
    return False


def analyze_tensor_factory_call(
    node: ast.Call,
    source_globals: Mapping[str, Any],
    receiver_is_tensor: bool,
    resolve_size_rank: Callable[[ast.expr], int | None],
    resolve_tensor_fact: Callable[[ast.expr], tuple[bool, int | None]],
) -> TensorFactoryFact | None:
    """Return a factory fact for a familiar call spelling, without executing it.

    `native=False` means the spelling was a candidate but callable identity or
    argument form was not established; callers should retain dependencies and
    report a warning instead of assigning tensor facts.
    """
    function = node.func
    name = (
        function.attr
        if isinstance(function, ast.Attribute)
        else function.id
        if isinstance(function, ast.Name)
        else ""
    )
    resolved = _callable_for(function, source_globals)
    alias_name = next(
        (factory_name for factory_name, native_factory in _FACTORIES.items() if resolved is native_factory),
        None,
    )
    if alias_name is not None:
        name = alias_name
    if name not in _CANDIDATES:
        return None

    if any(isinstance(argument, ast.Starred) for argument in node.args) or any(
        keyword.arg is None for keyword in node.keywords
    ):
        return TensorFactoryFact(name, False, reason="dynamic *args/**kwargs")

    if name in _METHOD_FACTORIES:
        if not isinstance(function, ast.Attribute) or not receiver_is_tensor:
            return TensorFactoryFact(name, False, reason="native tensor receiver was not established")
        native = name in _NATIVE_METHODS
    else:
        native = resolved is _FACTORIES[name]
    if not native:
        return TensorFactoryFact(name, False, reason="callable identity did not match native PyTorch")

    keyword_values = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}
    size_keywords = [keyword for keyword in node.keywords if keyword.arg == "size"]
    if len(size_keywords) > 1:
        return TensorFactoryFact(name, False, reason="duplicate size arguments")

    if name in _LIKE_FACTORIES:
        if len(node.args) > (2 if name == "full_like" else 1):
            return TensorFactoryFact(name, False, reason="too many positional arguments")
        input_node = node.args[0] if node.args else keyword_values.get("input")
        if input_node is None or (node.args and "input" in keyword_values):
            return TensorFactoryFact(name, False, reason="invalid like-factory arguments")
        has_fill = len(node.args) >= 2 or "fill_value" in keyword_values
        if name == "full_like" and not has_fill:
            return TensorFactoryFact(name, False, reason="missing fill_value")
        if name == "full_like" and len(node.args) == 2 and "fill_value" in keyword_values:
            return TensorFactoryFact(name, False, reason="duplicate fill_value arguments")
        if "size" in keyword_values:
            return TensorFactoryFact(name, False, reason="like factories do not accept size")
        input_is_tensor, rank = resolve_tensor_fact(input_node)
        if not input_is_tensor:
            return TensorFactoryFact(name, False, reason="tensor input identity was not established")
        return TensorFactoryFact(name, True, rank)

    if name in _METHOD_FACTORIES:
        if node.args:
            size_node = node.args[0]
            if "size" in keyword_values or len(node.args) > (2 if name == "new_full" else 1):
                return TensorFactoryFact(name, False, reason="invalid method-factory arguments")
        else:
            size_node = keyword_values.get("size")
        if size_node is None:
            return TensorFactoryFact(name, False, reason="missing size")
        if name == "new_full" and len(node.args) < 2 and "fill_value" not in keyword_values:
            return TensorFactoryFact(name, False, reason="missing fill_value")
        if name == "new_full" and len(node.args) == 2 and "fill_value" in keyword_values:
            return TensorFactoryFact(name, False, reason="duplicate fill_value arguments")
        if _known_invalid_size(size_node):
            return TensorFactoryFact(name, False, reason="invalid size")
        return TensorFactoryFact(name, True, _rank_of_size(size_node, resolve_size_rank))

    if name == "full" and len(node.args) < 2 and "fill_value" not in keyword_values:
        return TensorFactoryFact(name, False, reason="missing fill_value")
    if name == "full":
        if len(node.args) == 2 and "fill_value" in keyword_values:
            return TensorFactoryFact(name, False, reason="duplicate fill_value arguments")
        if len(node.args) > 2 or (node.args and "size" in keyword_values):
            return TensorFactoryFact(name, False, reason="invalid size arguments")
        size_node = node.args[0] if node.args else keyword_values.get("size")
        if not isinstance(size_node, (ast.Tuple, ast.List, ast.Name)):
            return TensorFactoryFact(name, False, reason="full requires a tuple or list size")
    else:
        if node.args and "size" in keyword_values:
            return TensorFactoryFact(name, False, reason="duplicate size arguments")
        if len(node.args) > 1:
            if any(
                isinstance(argument, (ast.Tuple, ast.List)) or _invalid_dimension(argument)
                for argument in node.args
            ):
                return TensorFactoryFact(name, False, reason="invalid size")
            return TensorFactoryFact(name, True, len(node.args))
        size_node = node.args[0] if node.args else keyword_values.get("size")
    if _known_invalid_size(size_node):
        return TensorFactoryFact(name, False, reason="invalid size")
    return TensorFactoryFact(name, True, _rank_of_size(size_node, resolve_size_rank))
