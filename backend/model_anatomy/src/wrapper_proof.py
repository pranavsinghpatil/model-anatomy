"""Bounded decorator passthrough guards for the explicit static analysis profile."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import CodeType, MethodType
from typing import Any

import torch

from .source_eval import ModuleState, SourceBoundary, SourceEvaluator


def nested_code(factory: Any, code: CodeType) -> bool:
    """Compare actual code identities, never a claimed callable name."""
    pending = [factory.__code__]
    while pending:
        item = pending.pop()
        if item is code:
            return True
        pending.extend(value for value in item.co_consts if isinstance(value, CodeType))
    return False


def forward_wrappers(function: Any, module: torch.nn.Module,
                     supplied_keywords: dict[str, Any]) -> dict[str, Any]:
    """Prove safe native profile wrappers or an unchanged tail-call with static guards."""
    from transformers.utils.deprecation import deprecate_kwarg
    from transformers.utils.generic import can_return_tuple, check_model_inputs

    records, failures = [], []
    for _ in range(8):
        target = getattr(function, "__wrapped__", None)
        if target is None:
            break
        try:
            text = textwrap.dedent(inspect.getsource(function.__code__))
            tree = next(item for item in ast.parse(text).body if isinstance(item, ast.FunctionDef))
            closure = inspect.getclosurevars(function).nonlocals
            record: dict[str, Any] = {"source": f"{function.__module__}.{function.__qualname__}",
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(), "conditions": [],
                "status": "unknown", "forward_executed": False}
            if nested_code(deprecate_kwarg, function.__code__):
                old_name = closure.get("old_name")
                if type(old_name) is not str or old_name in supplied_keywords:
                    raise SourceBoundary("Deprecated keyword translation is outside this recorded call profile")
                record.update(status="established", rule="native_keyword_alias_absent",
                              conditions=[{"keyword": old_name, "supplied": False}])
            elif nested_code(can_return_tuple, function.__code__) or nested_code(check_model_inputs, function.__code__):
                expected = {"use_cache": False, "output_attentions": False, "output_hidden_states": False, "return_dict": True}
                if module.training or any(name in supplied_keywords and supplied_keywords[name] is not value for name, value in expected.items()):
                    raise SourceBoundary("Native output/cache wrapper is outside the proven canonical profile")
                record.update(status="conditional", rule="native_output_capture_disabled_profile",
                    conditions=[{"use_cache": False, "output_attentions": False,
                                 "output_hidden_states": False, "return_dict": True,
                                 "training": False}],
                    scope="Explicit canonical profile; optional capture/cache branches are excluded")
            else:
                evaluator = SourceEvaluator()
                state = {name: value for name, value in vars(module).items()
                         if type(value) in (type(None), bool, int, float, str)}
                env = {"self": ModuleState(state, {"module_type": type(module).__qualname__})}
                positional = [item.arg for item in tree.args.posonlyargs + tree.args.args]

                def selected(nodes: list[ast.stmt]) -> bool:
                    for node in nodes:
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                            continue
                        if isinstance(node, ast.If):
                            condition = evaluator.eval(node.test, env, function.__globals__)
                            if type(condition) is not bool:
                                raise SourceBoundary("Decorator condition depends on unproven values")
                            record["conditions"].append({"condition": ast.unparse(node.test), "value": condition})
                            branch = node.body if condition else node.orelse
                            if branch and selected(branch):
                                return True
                        elif isinstance(node, ast.With) and len(node.items) == 1:
                            context = node.items[0].context_expr
                            factory = closure.get(context.func.id) if isinstance(context, ast.Call) and isinstance(context.func, ast.Name) else None
                            if (type(factory) is not MethodType or type(factory.__self__) is not torch.no_grad
                                    or getattr(factory, "__func__", None) is not torch.no_grad.clone
                                    or context.args or context.keywords or node.items[0].optional_vars):
                                raise SourceBoundary("Decorator context effects are unproven")
                            record["conditions"].append({"rule": "native_no_grad_context", "scope": "Forward tensor operations only; gradients are not represented"})
                            return selected(node.body)
                        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                            call = node.value
                            if not isinstance(call.func, ast.Name) or closure.get(call.func.id) is not target:
                                raise SourceBoundary("Decorator does not return the wrapped callable directly")
                            expected = [ast.Name(id=name, ctx=ast.Load()) for name in positional]
                            if tree.args.vararg:
                                expected.append(ast.Starred(value=ast.Name(id=tree.args.vararg.arg, ctx=ast.Load()), ctx=ast.Load()))
                            expected_kwargs = [ast.keyword(arg=None, value=ast.Name(id=tree.args.kwarg.arg, ctx=ast.Load()))] if tree.args.kwarg else []
                            if [ast.dump(item) for item in call.args] != [ast.dump(item) for item in expected] or [ast.dump(item) for item in call.keywords] != [ast.dump(item) for item in expected_kwargs]:
                                raise SourceBoundary("Decorator changes the wrapped arguments")
                            return True
                        else:
                            raise SourceBoundary(f"Selected decorator effects require proof: {type(node).__name__}")
                    return False

                if not selected(tree.body):
                    raise SourceBoundary("Decorator has no proven unchanged return path")
                record.update(status="established", rule="guarded_unchanged_tail_call")
            records.append(record)
        except (SourceBoundary, OSError, TypeError, AttributeError, SyntaxError, StopIteration, ValueError) as exc:
            failures.append(str(exc))
            break
        function = target
    else:
        failures.append("Decorator chain exceeds the analysis budget")
    return {"status": "unknown" if failures else "established", "wrappers": records,
            "failure_reasons": failures, "forward_executed": False}
