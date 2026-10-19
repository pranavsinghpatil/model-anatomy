"""Conservative computation-template evidence, without reading tensor contents."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import textwrap
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any

import torch

from .source_eval import SourceBoundary, SourceEvaluator


class EquivalenceEvidence:
    """Equal shapes are necessary but insufficient for folding independent modules."""

    def __init__(self) -> None:
        self.sources: dict[FunctionType, tuple[ast.FunctionDef, dict[str, str]]] = {}

    def source(self, function: Any) -> tuple[ast.FunctionDef, dict[str, str]]:
        function = inspect.unwrap(function)
        if type(function) is not FunctionType or any(name != "__class__" for name in function.__code__.co_freevars):
            raise SourceBoundary("Callable source or closure state cannot be established")
        if function not in self.sources:
            text = textwrap.dedent(inspect.getsource(function))
            if len(text) > 131072:
                raise SourceBoundary("Callable source exceeds the text budget")
            tree = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef))
            if sum(1 for _ in ast.walk(tree)) > 4096:
                raise SourceBoundary("Callable source exceeds the AST budget")
            self.sources[function] = tree, {"source": f"{function.__module__}.{function.__qualname__}",
                "sha256": hashlib.sha256(text.encode()).hexdigest()}
        return self.sources[function]

    def inspect(self, module: torch.nn.Module) -> dict[str, Any]:
        from .wrapper_proof import forward_wrappers
        wrapper_proof = forward_wrappers(type(module).forward, module, {})
        reads: dict[str, Any] = {}
        sources: dict[str, dict[str, str]] = {}
        failures: set[str] = set()
        failures.update(wrapper_proof["failure_reasons"])
        if inspect.getattr_static(type(module), "__getattribute__", object.__getattribute__) is not object.__getattribute__:
            failures.add("Custom module attribute lookup is not template-equivalent")
        if inspect.getattr_static(type(module), "__getattr__", torch.nn.Module.__getattr__) is not torch.nn.Module.__getattr__:
            failures.add("Custom module fallback lookup requires source proof")
        seen: set[tuple[int, int]] = set()

        def bounded(value: Any, depth: int = 0) -> Any:
            if depth > 6:
                raise SourceBoundary("State exceeds its nesting budget")
            if type(value) in (type(None), bool, int, str) and (type(value) is not str or len(value) <= 4096):
                return value
            if type(value) is float and math.isfinite(value):
                return value
            if type(value) is torch.dtype:
                return str(value)
            if type(value) in (list, tuple, set, frozenset) and len(value) <= 128:
                items = [bounded(item, depth + 1) for item in value]
                if type(value) in (set, frozenset):
                    items = sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
                return {"literal_type": type(value).__name__, "items": items}
            if type(value) is dict and len(value) <= 128 and all(type(key) is str for key in value):
                return {"literal_type": "dict", "items": [[key, bounded(item, depth + 1)] for key, item in value.items()]}
            raise SourceBoundary("State is not a bounded literal or verified callable")

        def resolve(owner: Any, names: list[str]) -> Any:
            for name in names:
                if isinstance(owner, torch.nn.Module):
                    if name in owner._modules:
                        owner = owner._modules[name]
                    elif name in owner._parameters:
                        owner = owner._parameters[name]
                    elif name in owner._buffers:
                        owner = owner._buffers[name]
                    elif name in vars(owner):
                        owner = vars(owner)[name]
                    else:
                        owner = inspect.getattr_static(owner, name)
                else:
                    # Native config aliases/properties are interpreted as bounded AST.
                    owner = SourceEvaluator().attribute(owner, name)
            return owner

        def visit(function: Any, owner: torch.nn.Module, aliases: set[str], depth: int = 0) -> None:
            if hasattr(function, "__wrapped__"):
                failures.add("Decorated helper dependency requires separate state proof")
                return
            if depth > 6 or len(seen) >= 64:
                failures.add("Helper dependency budget exceeded")
                return
            key = id(inspect.unwrap(function)), id(owner)
            if key in seen:
                return
            seen.add(key)
            try:
                tree, provenance = self.source(function)
            except (SourceBoundary, OSError, TypeError, SyntaxError, StopIteration, ValueError) as exc:
                failures.add(str(exc))
                return
            sources[provenance["source"]] = provenance
            aliases = set(aliases)
            assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
            for node in assignments:
                if isinstance(node.value, ast.Name) and node.value.id in aliases:
                    aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
            for node in assignments:
                if any(isinstance(target, ast.Name) and target.id in aliases for target in node.targets) and not (isinstance(node.value, ast.Name) and node.value.id in aliases):
                    failures.add("Module alias reassignment requires dataflow proof")
            parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Global, ast.Nonlocal)):
                    failures.add("Mutable global or closure state is not template-equivalent")
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in aliases:
                    global_value = inspect.unwrap(function).__globals__.get(node.id)
                    if type(global_value) in (dict, list, set):
                        failures.add(f"Mutable global dependency is unproven: {node.id}")
                    elif node.id in inspect.unwrap(function).__globals__ and not isinstance(global_value, (ModuleType, FunctionType, type)):
                        if isinstance(global_value, BuiltinFunctionType) and getattr(global_value, "__module__", "") in {"torch", "builtins", "math"}:
                            pass
                        elif any(secret in node.id.lower() for secret in ("secret", "password", "api_key", "access_token")):
                            failures.add("Credential-like global state is not recorded")
                        else:
                            try:
                                reads[f"global:{function.__module__}.{node.id}"] = {"kind": "global_literal", "value": bounded(global_value)}
                            except SourceBoundary:
                                failures.add(f"Opaque global dependency is unproven: {node.id}")
                if isinstance(node, ast.Attribute) and not isinstance(parents.get(id(node)), ast.Attribute):
                    parts, root = [], node
                    while isinstance(root, ast.Attribute):
                        parts.insert(0, root.attr)
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in aliases:
                        name = ".".join(parts)
                        try:
                            if any(part.lower() in {"token", "access_token", "password", "api_key", "secret"} for part in parts):
                                raise SourceBoundary("Credential-like state is not recorded")
                            value = resolve(owner, parts)
                            if isinstance(value, torch.nn.Parameter):
                                reads[name] = {"kind": "parameter_role", "shape": list(value.shape), "dtype": str(value.dtype)}
                            elif isinstance(value, torch.Tensor):
                                reads[name] = {"kind": "buffer_dependency", "shape": list(value.shape), "dtype": str(value.dtype)}
                                failures.add(f"Buffer contents/construction state unproven: {name}")
                            elif isinstance(value, torch.nn.Module):
                                reads[name] = {"kind": "child_module", "type": f"{type(value).__module__}.{type(value).__qualname__}"}
                            elif type(value) is FunctionType:
                                method_owner = resolve(owner, parts[:-1]) if len(parts) > 1 else owner
                                if isinstance(method_owner, torch.nn.Module):
                                    visit(value, method_owner, {"self"}, depth + 1)
                            elif isinstance(value, property):
                                raise SourceBoundary(f"Dynamic module property is unproven: {name}")
                            else:
                                reads[name] = {"kind": "literal_state", "value": bounded(value)}
                        except (SourceBoundary, AttributeError, TypeError, ValueError, RecursionError) as exc:
                            failures.add(f"{name}: {exc}")
                if isinstance(node, ast.Call):
                    namespace = node.func
                    while isinstance(namespace, ast.Attribute):
                        namespace = namespace.value
                    if isinstance(node.func, ast.Attribute) and isinstance(namespace, ast.Name):
                        namespace_value = inspect.unwrap(function).__globals__.get(namespace.id)
                        if isinstance(namespace_value, ModuleType):
                            failures.add(f"Namespace callable state requires separate proof: {ast.unparse(node.func)}")
                    if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "setattr", "eval", "exec"}:
                        failures.add(f"Dynamic state access requires separate proof: {node.func.id}")
                    if isinstance(node.func, ast.Name):
                        target = inspect.unwrap(function).__globals__.get(node.func.id)
                        if target is None and node.func.id not in {"len", "range", "tuple", "list", "dict", "set", "float", "int", "bool", "str", "isinstance", "hasattr", "super", "max", "min", "abs", "enumerate", "zip"}:
                            failures.add(f"Local/dynamic callable dependency is unproven: {node.func.id}")
                        if type(target) is FunctionType:
                            params = inspect.signature(target).parameters
                            module_argument = next((i for i, arg in enumerate(node.args) if isinstance(arg, ast.Name) and arg.id in aliases), None)
                            target_aliases = {list(params)[module_argument]} if module_argument is not None and module_argument < len(params) else set()
                            target_aliases.update(keyword.arg for keyword in node.keywords if keyword.arg is not None and isinstance(keyword.value, ast.Name) and keyword.value.id in aliases)
                            visit(target, owner, target_aliases, depth + 1)

        if "forward" in vars(module) or module._forward_hooks or module._forward_pre_hooks:
            failures.add("Instance forward override or hooks are not template-equivalent")
        containers = (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
        if type(module) not in containers:
            visit(inspect.unwrap(type(module).forward), module, {"self"})
            if isinstance(module, containers):
                for name in ("__iter__", "__getitem__"):
                    method = getattr(type(module), name, None)
                    if method is not None:
                        visit(method, module, {"self"})
        constructor, constructor_state = None, {}
        try:
            tree, provenance = self.source(type(module).__init__)
            constructor = dict(provenance, buffer_statements=[ast.unparse(node) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register_buffer"])
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                    if node.attr in vars(module) and not any(secret in node.attr.lower() for secret in ("secret", "password", "api_key", "access_token")):
                        try:
                            constructor_state[node.attr] = bounded(vars(module)[node.attr])
                        except SourceBoundary:
                            pass
        except (SourceBoundary, OSError, TypeError, SyntaxError, StopIteration, ValueError) as exc:
            if module._buffers:
                failures.add(f"Buffer constructor unavailable: {exc}")
        if module._buffers:
            failures.add("Independent buffer values cannot be equated from shape/dtype or constructor text")
        return {"status": "unknown" if failures else "established", "state_reads": reads,
                "forward_wrapper_proof": wrapper_proof,
                "sources": sorted(sources.values(), key=lambda item: item["source"]),
                "buffer_constructor": constructor, "failures": sorted(failures),
                "constructor_state": constructor_state,
                "scope": "Computation template and literal state; learned parameter values are intentionally not equated",
                "tensor_values_read": False, "forward_executed": False}
