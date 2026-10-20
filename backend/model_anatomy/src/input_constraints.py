"""Recognize bounded input preconditions from pure, fail-fast source guards."""

from __future__ import annotations

import ast
from typing import Any, Callable


def dimension_guard(node: ast.If, symbol: Callable[[ast.AST], Any]) -> tuple[str, ast.AST, ast.AST] | None:
    """Identify an equality required to avoid an unconditional builtin exception."""
    if node.orelse or len(node.body) != 1 or not isinstance(node.body[0], ast.Raise):
        return None
    raised = node.body[0]
    if raised.cause is not None or not isinstance(raised.exc, ast.Call):
        return None
    if not any(symbol(raised.exc.func) is exception for exception in (ValueError, TypeError, AssertionError, RuntimeError)):
        return None
    if raised.exc.keywords or any(isinstance(item, (ast.Call, ast.NamedExpr, ast.Await))
                                 for arg in raised.exc.args for item in ast.walk(arg)):
        return None
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.NotEq):
        return None
    for dimension, expected in ((test.left, test.comparators[0]), (test.comparators[0], test.left)):
        if (isinstance(dimension, ast.Subscript) and isinstance(dimension.value, ast.Attribute)
                and dimension.value.attr == "shape" and isinstance(dimension.value.value, ast.Name)):
            if any(isinstance(item, (ast.Call, ast.NamedExpr, ast.Await))
                   for expression in (dimension.slice, expected) for item in ast.walk(expression)):
                continue
            return dimension.value.value.id, dimension.slice, expected
    return None
