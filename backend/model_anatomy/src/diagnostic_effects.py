"""Prove a restricted helper has only local string and native logging effects."""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import textwrap
from typing import Any

from transformers.utils import logging as native_logging
from transformers.utils import import_utils
from transformers.utils.import_utils import is_torch_fx_proxy, is_torchdynamo_compiling
import torch

from .source_eval import ModuleState, SourceBoundary, SourceEvaluator, Symbol, Tensor

_WARNING_ONCE = native_logging.warning_once


def diagnostic_effects(function: Any, owner: ModuleState, arguments: dict[str, Any]) -> dict[str, Any] | None:
    """Reject mutation, opaque calls, custom loggers and non-None returns conservatively."""
    try:
        text = textwrap.dedent(inspect.getsource(function))
        tree = next(item for item in ast.parse(text).body if isinstance(item, ast.FunctionDef))
        if tree.decorator_list or sum(1 for _ in ast.walk(tree)) > 1024:
            return None
        evaluator = SourceEvaluator()
        env = {"self": owner, **arguments}
        globals = function.__globals__
        calls, conditions, assumptions = [], [], []

        def value(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                if type(node.value) not in (type(None), bool, int, float, str):
                    raise SourceBoundary("Unsupported diagnostic literal")
                return node.value
            if isinstance(node, (ast.Name, ast.Attribute)):
                result = evaluator.eval(node, env, globals)
                if not isinstance(result, (Tensor, Symbol, ModuleState)) and type(result) not in (type(None), bool, int, float, str):
                    raise SourceBoundary("Diagnostic expression contains an opaque object")
                return result
            if isinstance(node, (ast.Tuple, ast.List)):
                return tuple(value(part) for part in node.elts)
            if isinstance(node, ast.Slice):
                return tuple(value(part) if part else None for part in (node.lower, node.upper, node.step))
            if isinstance(node, ast.Subscript):
                base = value(node.value)
                if not isinstance(base, Tensor):
                    raise SourceBoundary("Diagnostic indexing is not a native tensor read")
                value(node.slice)
                return Tensor((), base.dtype, "diagnostic_tensor_read")
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
                operand = value(node.operand)
                if not isinstance(operand, (Tensor, Symbol)) and type(operand) not in (bool, int, float):
                    raise SourceBoundary("Diagnostic unary operand is opaque")
                return Symbol("diagnostic_condition")
            if isinstance(node, (ast.BoolOp, ast.Compare)):
                parts = node.values if isinstance(node, ast.BoolOp) else [node.left, *node.comparators]
                for part in parts:
                    value(part)
                return Symbol("diagnostic_condition")
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if isinstance(part, ast.FormattedValue):
                        if part.format_spec is not None or type(value(part.value)) not in (type(None), bool, int, float, str):
                            raise SourceBoundary("Diagnostic formatting contains an opaque value")
                    else:
                        value(part)
                return "<diagnostic string>"
            if isinstance(node, ast.Call):
                target = evaluator.eval(node.func, env, globals)
                if not any(target is predicate for predicate in (is_torch_fx_proxy, torch.jit.is_tracing, is_torchdynamo_compiling, getattr(import_utils, "is_tracing", None))):
                    raise SourceBoundary("Diagnostic guard contains an unverified call")
                if len(node.args) != (1 if target is is_torch_fx_proxy or target is getattr(import_utils, "is_tracing", None) else 0):
                    raise SourceBoundary("Diagnostic predicate signature is invalid")
                for argument in node.args:
                    value(argument)
                if node.keywords:
                    raise SourceBoundary("Diagnostic predicate keywords are unsupported")
                assumptions.append({"predicate": f"{target.__module__}.{target.__qualname__}",
                                    "source_sha256": hashlib.sha256(inspect.getsource(target).encode()).hexdigest(),
                                    "value": False, "basis": "Canonical native tensor profile; forward, FX/JIT tracing and compilation are disabled"})
                return Symbol("diagnostic_condition")
            raise SourceBoundary("Diagnostic expression effects are unsupported")

        def statements(nodes: list[ast.stmt]) -> None:
            for node in nodes:
                if isinstance(node, ast.If):
                    value(node.test)
                    conditions.append({"condition": ast.unparse(node.test), "line": node.lineno,
                                       "evaluation": "Not executed; both paths have restricted diagnostic effects"})
                    statements(node.body)
                    statements(node.orelse)
                elif isinstance(node, ast.Return) and (node.value is None or isinstance(node.value, ast.Constant) and node.value.value is None):
                    continue
                elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                    result = value(node.value)
                    if type(result) is not str:
                        raise SourceBoundary("Only local diagnostic strings may be assigned")
                    for target in node.targets:
                        if target.id in arguments or target.id == "self":
                            raise SourceBoundary("Diagnostic input rebinding is unsupported")
                        env[target.id] = result
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
                    if type(env.get(node.target.id)) is not str or type(value(node.value)) is not str:
                        raise SourceBoundary("Diagnostic in-place effects are not proven to be local strings")
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str:
                    continue
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    call = node.value
                    logger = evaluator.eval(call.func.value, env, globals)
                    target = inspect.getattr_static(type(logger), call.func.attr, None)
                    if type(logger) is not logging.Logger or target is not _WARNING_ONCE:
                        raise SourceBoundary("Diagnostic logging target is not the native logger")
                    for argument in call.args:
                        value(argument)
                    if call.keywords:
                        raise SourceBoundary("Diagnostic logging keywords are unsupported")
                    calls.append({"call": ast.unparse(call.func), "line": call.lineno,
                                  "source_sha256": hashlib.sha256(inspect.getsource(inspect.unwrap(target)).encode()).hexdigest()})
                else:
                    raise SourceBoundary(f"Diagnostic statement effects are unsupported: {type(node).__name__}")

        statements(tree.body)
        if not calls:
            return None
        return {"rule": "restricted_diagnostic_effects", "source": f"{function.__module__}.{function.__qualname__}",
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(), "conditions": conditions,
                "logging_calls": calls, "module_state_mutation": False, "tensor_mutation": False,
                "profile_assumptions": assumptions,
                "returns": None, "executed": False,
                "limitation": "Diagnostic content and data-dependent conditions are not evaluated; runtime input validity is not established"}
    except (SourceBoundary, AttributeError, TypeError, ValueError, OSError, SyntaxError, StopIteration):
        return None
