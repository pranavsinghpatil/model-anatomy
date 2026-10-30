"""Bounded module-call dependencies, without interpreting tensor expressions.

Only registered module references, literal configuration guards and container
iteration are resolved. Other expressions carry dependencies, never tensor values
or claimed operations. Python source is read, never executed.
"""

from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import inspect
import itertools
import math
import operator
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from types import CodeType, MappingProxyType, ModuleType, UnionType
from typing import Any

import torch

from . import limits
from .callable_metadata import (
    MISSING,
    AttentionInterface,
    ExpertsInterface,
    attention_dispatch,
    static_attribute,
)
from .module_primitives import functional_primitive, primitive_proof, script_primitive_proof

UNKNOWN = object()


@dataclass(frozen=True)
class Value:
    module: str | None = None
    method: str | None = None
    deferred: bool = False
    unordered: bool = False
    function: Any = field(default=None, repr=False, compare=False)
    namespace: Any = field(default=None, repr=False, compare=False)
    literal: Any = field(default=UNKNOWN, repr=False)
    origins: frozenset[str] = frozenset()
    origin_guards: tuple[tuple[str, tuple[str, ...]], ...] = ()
    items: tuple[Value, ...] | None = None
    config: Any = field(default=None, repr=False, compare=False)
    mapping: dict[Any, Value] | None = field(default=None, repr=False, compare=False)
    primitive: type | None = field(default=None, repr=False, compare=False)
    factory: bool = False
    raises: bool = False
    generator: Any = field(default=None, repr=False, compare=False)
    element: Value | None = field(default=None, repr=False, compare=False)
    local_function: Any = field(default=None, repr=False, compare=False)
    intrinsic: Any = field(default=None, repr=False, compare=False)
    index: str | None = None
    choices: tuple[Value, ...] | None = None
    selection: str | None = None
    conditional_items: bool = False
    virtual: Any = field(default=None, repr=False, compare=False)
    bound_self: Any = field(default=None, repr=False, compare=False)
    script_proof: Any = field(default=None, repr=False, compare=False)
    local_defaults: Any = field(default=None, repr=False, compare=False)
    mutable: bool = False
    optional_keys: frozenset[Any] = frozenset()
    type_object: Any = field(default=None, repr=False, compare=False)
    registry: bool = False


def has_module_reference(value: Value) -> bool:
    return (value.module is not None or value.deferred or value.function is not None
            or value.local_function is not None
            or value.virtual is not None
            or bool(value.choices and any(has_module_reference(item) for item in value.choices))
            or bool(value.items and any(has_module_reference(item) for item in value.items))
            or bool(value.mapping and any(has_module_reference(item) for item in value.mapping.values())))


def dependencies(values: list[Value]) -> Value:
    return Value(origins=frozenset().union(*(v.origins for v in values)),
                 deferred=any(has_module_reference(v) for v in values),
                 origin_guards=tuple(dict.fromkeys(guard for value in values for guard in value.origin_guards)),
                 raises=any(value.raises for value in values))


def merge(left: Value, right: Value) -> Value:
    if left is right:
        return left
    items = tuple(merge(a, b) for a, b in zip(left.items, right.items)) if left.items is not None and right.items is not None and len(left.items) == len(right.items) else None
    mapping = {key: merge(left.mapping.get(key, Value()), right.mapping.get(key, Value()))
               for key in left.mapping.keys() | right.mapping.keys()} if left.mapping is not None and right.mapping is not None else None
    optional = (left.optional_keys | right.optional_keys | (left.mapping.keys() ^ right.mapping.keys())) if mapping is not None else frozenset()
    return Value(module=left.module if left.module == right.module else None,
                 method=left.method if left.method == right.method else None,
                 deferred=has_module_reference(left) or has_module_reference(right),
                 function=left.function if left.function is right.function else None,
                 items=items, mapping=mapping, optional_keys=frozenset(optional), mutable=left.mutable and right.mutable,
                 primitive=left.primitive if left.primitive is right.primitive else None,
                 factory=left.factory and right.factory,
                 registry=left.registry or right.registry,
                 origins=left.origins | right.origins,
                 origin_guards=tuple(dict.fromkeys((*left.origin_guards, *right.origin_guards))),
                 literal=left.literal if type(left.literal) in (bool, int, str, type(None))
                 and type(left.literal) is type(right.literal) and left.literal == right.literal else UNKNOWN)


class ModuleTopology:
    """Collect source-backed relationships while retaining unsupported control flow."""

    def __init__(self, model: torch.nn.Module, raw: dict[str, Any], analyzer: Any):
        self.model, self.raw, self.analyzer = model, raw, analyzer
        self.edges: list[dict[str, Any]] = []
        self.calls: dict[str, list[dict[str, Any]]] = {}
        self.reports: dict[str, dict[str, Any]] = {}
        self.steps = 0
        self.path = ""
        self.conditions: list[str] = []
        self.location: dict[str, Any] = {}
        self.active: set[tuple[Any, ...]] = set()
        self.current_report: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        self.loop_effects: list[bool] = []
        self.sources: dict[Any, tuple[Any, str, int]] = {}
        self.module_paths = {id(self.module(path)): path for path in reversed(list(raw["modules"]))}

    def inspect(self) -> None:
        for path, record in self.raw["modules"].items():
            if self.analyzer.is_leaf(path) or record["container"]:
                continue
            self.path, self.steps = path, 0
            self.current_report = {"status": "source_read", "boundaries": [], "conditions": [],
                                   "scope": "Module calls and dependencies only; tensor expressions omitted",
                                   "input_profile": {"public_arguments": "symbolic", "additional_keyword_arguments": "absent",
                                                     "module_training": vars(self.module(path)).get("training")}}
            self.reports[path] = self.current_report
            self.method(path, "forward", [], {}, root=True)
        # Sequential is an execution contract; ModuleList registration is not.
        for path, record in self.raw["modules"].items():
            if record["container"] == "sequential":
                for left, right in zip(record["children"], record["children"][1:]):
                    self.edges.append({"source": left, "target": right, "conditions": [],
                                       "basis": "torch.nn.Sequential composition", "owner": path})

    def boundary(self, node: ast.AST | None, reason: str, *, blocking: bool = True,
                 category: str | None = None, **details: Any) -> None:
        item = {**self.location, "reason": reason, "statement": type(node).__name__ if node else None,
                "line": self.location.get("start_line", 1) + getattr(node, "lineno", 1) - 1,
                "conditions": list(self.conditions), "blocking": blocking,
                "category": category or ("architecture_gap" if blocking else "computation_detail"),
                **details}
        if item not in self.current_report["boundaries"]:
            self.current_report["boundaries"].append(item)
        if blocking:
            self.current_report["status"] = "partial_topology"

    def evidence(self, node: ast.AST) -> dict[str, Any]:
        return {**self.location, "line": self.location.get("start_line", 1) + getattr(node, "lineno", 1) - 1,
                "conditions": list(self.conditions), "owner": self.path,
                "basis": "Static module-call dependency; intermediate expressions are omitted"}

    def module(self, path: str) -> torch.nn.Module:
        return self.model.get_submodule(path) if path else self.model

    def value(self, obj: Any) -> Value:
        if any(obj is function for function in (math.log2, math.ceil, math.floor, math.sqrt)):
            return Value(intrinsic=obj)
        if proof := script_primitive_proof(obj):
            return Value(script_proof=proof)
        if type(obj) is functools._lru_cache_wrapper and inspect.isfunction(obj.__wrapped__):
            return Value(function=obj.__wrapped__)
        if obj is torch.nn.Parameter:
            return Value(intrinsic=obj)
        if obj is itertools.groupby:
            return Value(intrinsic=obj)
        if isinstance(obj, torch.nn.Module):
            path = self.module_paths.get(id(obj))
            return Value(module=path) if path is not None else Value(primitive=primitive) if (primitive := functional_primitive(obj)) else Value(deferred=True)
        if inspect.ismethod(obj) and isinstance(obj.__self__, torch.nn.Module):
            path = self.module_paths.get(id(obj.__self__))
            return Value(module=path, method=obj.__name__) if path is not None else Value(deferred=True)
        if inspect.ismethod(obj):
            return Value(function=obj.__func__, bound_self=Value(namespace=obj.__self__))
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
            if primitive := functional_primitive(obj):
                return Value(primitive=primitive, factory=True, type_object=obj)
            if obj.__module__.startswith(("torch.nn.modules.", "transformers.")) and obj.__bases__ == (torch.nn.Module,):
                return Value(virtual=(obj, {}), factory=True, type_object=obj)
            return Value(deferred=True, type_object=obj)
        if isinstance(obj, type):
            return Value(type_object=obj)
        if inspect.isfunction(obj):
            return Value(function=obj)
        if type(obj) in (list, tuple) and len(obj) <= limits.MAX_MODEL_LAYERS:
            return Value(items=tuple(self.value(item) for item in obj))
        if type(obj) in (dict, OrderedDict, MappingProxyType) and len(obj) <= limits.MAX_MODEL_LAYERS:
            return Value(mapping={key: self.value(item) for key, item in obj.items() if type(key) in (str, int)})
        if isinstance(obj, ModuleType) or type(obj) in (AttentionInterface, ExpertsInterface):
            return Value(namespace=obj)
        if type(obj) in (int, float, bool, str, type(None)):
            return Value(literal=obj)
        if callable(obj) and not isinstance(obj, type) and not inspect.isbuiltin(obj):
            return Value(deferred=True)
        return Value()

    def name(self, name: str, env: dict[str, Value]) -> Value:
        return env[name] if name in env else self.value(self.globals.get(name, getattr(builtins, name, MISSING)))

    def attribute(self, base: Value, name: str) -> Value:
        if base.raises:
            return base
        if base.choices is not None:
            return Value(choices=tuple(replace(self.attribute(item, name), selection=item.selection) for item in base.choices))
        if base.virtual is not None:
            implementation, fields = base.virtual
            if name == "__class__":
                return Value(virtual=(implementation, {}), factory=True)
            if base.factory and name in {"__name__", "__qualname__", "__module__"}:
                return Value(literal=getattr(implementation, name))
            if name in fields:
                return fields[name]
            descriptor = inspect.getattr_static(implementation, name, MISSING)
            if isinstance(descriptor, staticmethod):
                return Value(function=descriptor.__func__)
            if inspect.isfunction(descriptor):
                return replace(base, method=name, factory=False)
            if type(descriptor) in (str, int, bool, float, type(None)):
                return self.value(descriptor)
            return Value(deferred=True)
        if base.module is not None:
            module = self.module(base.module)
            if name == "__class__":
                return Value(namespace=type(module))
            if name == "__dict__":
                value = self.value(vars(module))
                if value.mapping is not None:
                    return replace(value, mapping={key: replace(item, registry=True) if key in {'_modules', '_parameters', '_buffers'} else item
                                                   for key, item in value.mapping.items()})
                return value
            if name in {'_modules', '_parameters', '_buffers'}:
                return replace(self.value(vars(module).get(name, {})), registry=True)
            if name in module._modules:
                return Value(module=f"{base.module}.{name}".lstrip(".")) if module._modules[name] is not None else Value(literal=None)
            if name == "config":
                return Value(config=vars(module).get("config"))
            if name in module._parameters or name in module._buffers:
                return Value()
            stored = static_attribute(module, name)
            if stored is not MISSING:
                return self.value(stored)
            descriptor = inspect.getattr_static(type(module), name, MISSING)
            if inspect.isfunction(descriptor):
                return Value(module=base.module, method=name)
            if isinstance(descriptor, staticmethod):
                return Value(function=descriptor.__func__)
            if type(descriptor) in (str, int, bool, float, tuple, list, dict, set, frozenset, type(None)):
                return self.value(descriptor)
            if descriptor is MISSING and inspect.getattr_static(type(module), "__getattr__") is torch.nn.Module.__getattr__:
                self.boundary(None, "Attribute absent from the constructed module; this source path raises",
                              blocking=False, category="invalid_input_path", module_path=base.module, attribute=name)
                return Value(raises=True)
            return Value(deferred=True)
        if base.config is not None:
            return self.value(static_attribute(base.config, name))
        if base.function is not None and name in {'__code__', '__name__', '__qualname__', '__module__'}:
            value = getattr(base.function, name)
            return Value(namespace=value) if type(value) is CodeType else self.value(value)
        if base.namespace is not None:
            if isinstance(base.namespace, ModuleType):
                return self.value(vars(base.namespace).get(name, MISSING))
            if type(base.namespace) in (AttentionInterface, ExpertsInterface):
                return Value(namespace=base.namespace, method=name)
            if type(base.namespace) is CodeType and name in {'co_varnames', 'co_argcount', 'co_kwonlyargcount', 'co_posonlyargcount'}:
                return self.value(getattr(base.namespace, name))
            stored = inspect.getattr_static(base.namespace, name, MISSING)
            if inspect.isfunction(stored):
                return Value(function=stored, bound_self=base)
            if name == "__class__":
                return Value(namespace=type(base.namespace))
            if isinstance(base.namespace, type) and name in {"__name__", "__qualname__", "__module__"}:
                return Value(literal=getattr(base.namespace, name))
            return self.value(stored) if stored is not MISSING else Value(deferred=True)
        return Value(origins=base.origins, origin_guards=base.origin_guards, deferred=self.capability(base))

    def reference(self, node: ast.AST, env: dict[str, Value]) -> Value:
        """Inspect capability references without recording or evaluating calls."""
        if isinstance(node, ast.Name):
            return self.name(node.id, env)
        if isinstance(node, ast.Attribute):
            return self.attribute(self.reference(node.value, env), node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            return Value(items=tuple(self.reference(item, env) for item in node.elts))
        if isinstance(node, ast.Subscript):
            return self.reference(node.value, env)
        return Value()

    def capability(self, value: Value) -> bool:
        return has_module_reference(value)

    def truth(self, value: Value) -> bool | None:
        """Resolve Python truth only from bounded metadata, never from tensors."""
        if type(value.literal) in (bool, int, float, str, type(None)):
            return bool(value.literal)
        if value.mapping is not None:
            if value.mapping.keys() - value.optional_keys:
                return True
            return False if not value.mapping else None
        if value.items is not None and not value.conditional_items:
            return bool(value.items)
        if value.function is not None or value.local_function is not None:
            return True
        return None

    def affects_modules(self, node: ast.AST, env: dict[str, Value]) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                target = self.reference(child.func, env)
                if target.function is inspect.signature:
                    continue
                if target.module is not None and target.method in {
                    "parameters", "named_parameters", "buffers", "named_buffers", "modules", "named_modules", "children", "named_children"
                } and inspect.getattr_static(type(self.module(target.module)), target.method) is inspect.getattr_static(torch.nn.Module, target.method):
                    continue
                target_effect = (target if target.function is None or self.function_captures_modules(target.function) else Value())
                values = [target_effect, *[self.reference(arg, env) for arg in child.args],
                          *[self.reference(kw.value, env) for kw in child.keywords]]
                if any(self.capability(value) for value in values):
                    return True
        return False

    def method(self, path: str, name: str, args: list[Value], kwargs: dict[str, Value], *, root: bool = False,
               unknown_star: bool = False) -> Value:
        function = inspect.getattr_static(type(self.module(path)), name, None)
        if name == '__getattr__' and function is torch.nn.Module.__getattr__:
            if args and type(args[0].literal) is str:
                return self.attribute(Value(module=path), args[0].literal)
            self.boundary(None, "Native module attribute name not statically established")
            return Value(deferred=True)
        inventory = {"parameters": "parameters", "named_parameters": "parameters",
                     "buffers": "buffers", "named_buffers": "buffers",
                     "children": "modules", "named_children": "modules", "modules": "modules", "named_modules": "modules"}
        if name in inventory and function is inspect.getattr_static(torch.nn.Module, name):
            prefix = path + "." if path else ""
            names = [item for item in self.raw[inventory[name]] if item == path or item.startswith(prefix)]
            if name in {"children", "named_children"}:
                names = self.raw["modules"][path]["children"]
            values = [Value(module=item) if inventory[name] == "modules" else Value() for item in names]
            if name.startswith("named_"):
                values = [Value(items=(Value(literal=item.removeprefix(prefix)), value)) for item, value in zip(names, values)]
            self.current_report.setdefault("metadata_reads", []).append({**self.location, "module_path": path,
                "method": name, "basis": "Native nn.Module inventory method identity; no forward invocation"})
            return Value(items=tuple(values))
        if root:
            args, kwargs = [], {}
            parameters = list(inspect.signature(function).parameters.values())
            for parameter in parameters[1:]:
                value = Value(origins=frozenset({f"@{path}:{parameter.name}"}))
                if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
                    args.append(value)
                elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwargs[parameter.name] = value
        return self.read_callable(function, [Value(module=path), *args], kwargs, root_path=path if root else None,
                                  unknown_star=unknown_star)

    def read_callable(self, function: Any, args: list[Value], kwargs: dict[str, Value], *,
                      root_path: str | None = None, unknown_star: bool = False) -> Value:
        key = (id(function), tuple((v.module, v.method) for v in args if v.module is not None))
        if key in self.active or len(self.active) >= 8:
            self.boundary(None, "Recursive or deeply nested module helper; calls not established")
            return dependencies([*args, *kwargs.values()])
        try:
            source_key = function.__code__
            if source_key not in self.sources:
                lines, line = inspect.getsourcelines(source_key)
                source = textwrap.dedent("".join(lines))
                tree = ast.parse(source).body[0]
                if not isinstance(tree, ast.FunctionDef):
                    raise TypeError("Not a Python function")
                self.sources[source_key] = tree, source, line
            tree, source, line = self.sources[source_key]
        except (OSError, TypeError, ValueError, SyntaxError, AttributeError):
            self.boundary(None, "Source unavailable for a callable receiving module references")
            return dependencies([*args, *kwargs.values()])
        previous_location, previous_globals = self.location, self.globals
        self.location = {"method": f"{function.__module__}.{function.__qualname__}",
                         "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "start_line": line,
                         "source_module": function.__globals__.get('__name__')}
        self.globals = dict(function.__globals__)
        try:
            self.globals.update(inspect.getclosurevars(function).nonlocals)
        except (TypeError, ValueError):
            pass
        parameters = [*tree.args.posonlyargs, *tree.args.args]
        defaults = dict(zip([p.arg for p in parameters[-len(tree.args.defaults):]], tree.args.defaults)) if tree.args.defaults else {}
        env: dict[str, Value] = {}
        for index, parameter in enumerate(parameters):
            if index < len(args):
                value = args[index]
            elif parameter.arg in kwargs:
                value = kwargs[parameter.arg]
            elif not unknown_star and parameter.arg in defaults:
                value = self.expression(defaults[parameter.arg], env)
            else:
                value = Value(deferred=unknown_star)
            env[parameter.arg] = value
        for parameter, default in zip(tree.args.kwonlyargs, tree.args.kw_defaults):
            env[parameter.arg] = (kwargs[parameter.arg] if parameter.arg in kwargs else
                                  self.expression(default, env) if default else Value(deferred=unknown_star))
        if tree.args.vararg:
            env[tree.args.vararg.arg] = Value(items=None if unknown_star else tuple(args[len(parameters):]), deferred=unknown_star,
                                            origins=dependencies(args[len(parameters):]).origins)
        if tree.args.kwarg:
            extras = {key: value for key, value in kwargs.items() if key not in env and key is not None}
            env[tree.args.kwarg.arg] = Value(origins=dependencies(list(extras.values())).origins,
                                            mapping=extras, deferred=any(self.capability(value) for value in extras.values()))
        self.active.add(key)
        try:
            result, flow = self.statements(tree.body, env)
            if root_path is not None and result.generator is not None and result.deferred:
                self.boundary(tree, "Forward returns deferred module calls without an observed consumer")
            return replace(result, raises=True) if flow == "raise" else result
        finally:
            self.active.remove(key)
            self.location, self.globals = previous_location, previous_globals

    def function_captures_modules(self, function: Any) -> bool:
        def contains_module(obj: Any, depth: int = 0) -> bool:
            if isinstance(obj, torch.nn.Module) or (inspect.ismethod(obj) and isinstance(obj.__self__, torch.nn.Module)):
                return True
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                return True
            if depth < 4 and type(obj) in (list, tuple, dict) and len(obj) <= limits.MAX_MODEL_LAYERS:
                values = obj.values() if type(obj) is dict else obj
                return any(contains_module(value, depth + 1) for value in values)
            return False
        try:
            closure = inspect.getclosurevars(function)
            if any(contains_module(value) for value in [*closure.nonlocals.values(), *closure.globals.values()]):
                return True
            namespace = {**function.__globals__, **closure.nonlocals}

            def resolve(node: ast.AST) -> Any:
                if isinstance(node, ast.Name):
                    return namespace.get(node.id)
                if isinstance(node, ast.Attribute) and isinstance(owner := resolve(node.value), ModuleType):
                    return vars(owner).get(node.attr)
                return None

            source = ast.parse(textwrap.dedent(''.join(inspect.getsourcelines(function.__code__)[0])))
            return any(contains_module(target) or target is torch.nn.Parameter for node in ast.walk(source)
                       if isinstance(node, ast.Call) for target in [resolve(node.func)])
        except (OSError, SyntaxError, AttributeError, TypeError, ValueError):
            return False

    def callable_hash(self, function: Any) -> str | None:
        entry = self.sources.get(function.__code__)
        return hashlib.sha256(entry[1].encode()).hexdigest() if entry else None

    def sequence(self, value: Value) -> tuple[Value, ...] | None:
        if value.generator is not None:
            node, environment = value.generator
            return self.comprehension(node, dict(environment)).items
        if value.choices is not None:
            sequences = [self.sequence(choice) for choice in value.choices]
            if sequences and all(items is not None for items in sequences):
                length = max(map(len, sequences))
                if sum(map(len, sequences)) <= limits.MAX_MODEL_LAYERS:
                    return tuple(Value(choices=tuple(replace(items[index], selection=choice.selection)
                                                    for choice, items in zip(value.choices, sequences) if index < len(items)))
                                 for index in range(length))
                self.boundary(None, "Finite container alternatives exceed module analysis budget")
            return None
        if value.items is not None:
            return value.items
        if value.mapping is not None:
            return tuple(Value(literal=key) for key in value.mapping)
        if value.module is not None:
            record = self.raw["modules"][value.module]
            if record["container"] in {"module_list", "sequential"}:
                return tuple(Value(module=p) for p in record["children"])
        return None

    def comprehension(self, node: ast.AST, env: dict[str, Value], *, first_only: bool = False) -> Value:
        outputs: list[Value] = []
        entries: dict[Any, Value] = {}
        uncertain = False

        def visit(index: int, local: dict[str, Value]) -> None:
            nonlocal uncertain
            if first_only and outputs:
                return
            if len(outputs) >= limits.MAX_MODEL_LAYERS or self.steps > 40_000:
                self.boundary(node, "Comprehension module analysis budget exceeded")
                return
            if index == len(node.generators):
                if isinstance(node, ast.DictComp):
                    key, value = self.expression(node.key, local), self.expression(node.value, local)
                    if type(key.literal) in (str, int):
                        entries[key.literal] = value
                    else:
                        uncertain = True
                else:
                    value = self.expression(node.elt, local)
                outputs.append(value)
                return
            generator = node.generators[index]
            iterable = self.expression(generator.iter, local)
            sequence = self.sequence(iterable)
            if sequence is None:
                uncertain = True
                sequence = (iterable.element or Value(),)
            for item in sequence:
                branch = dict(local)
                self.assign(generator.target, item, branch)
                depth = len(self.conditions)
                self.conditions.append("comprehension element is consumed")
                accept = True
                for condition in generator.ifs:
                    result = self.expression(condition, branch)
                    truth = self.truth(result)
                    if truth is False:
                        accept = False
                        break
                    if truth is not True:
                        uncertain = True
                        self.conditions.append(ast.unparse(condition)[:240])
                if accept:
                    visit(index + 1, branch)
                del self.conditions[depth:]

        visit(0, env)
        result = dependencies(outputs)
        element = outputs[0] if outputs else Value()
        for item in outputs[1:]:
            element = merge(element, item)
        return replace(result, items=None if uncertain or isinstance(node, ast.DictComp) else tuple(outputs),
                       mapping=entries if isinstance(node, ast.DictComp) and not uncertain else None,
                       element=element)

    def local_callable(self, tree: ast.AST, env: dict[str, Value]) -> Value:
        parameters = [*tree.args.posonlyargs, *tree.args.args]
        defaults = {parameter.arg: self.expression(default, env)
                    for parameter, default in zip(parameters[-len(tree.args.defaults):], tree.args.defaults)} if tree.args.defaults else {}
        defaults.update({parameter.arg: self.expression(default, env)
                         for parameter, default in zip(tree.args.kwonlyargs, tree.args.kw_defaults) if default is not None})
        closure = dict(env)
        value = Value(local_function=(tree, closure, dict(self.location), dict(self.globals)), local_defaults=defaults)
        if isinstance(tree, ast.FunctionDef):
            closure[tree.name] = value
        return value

    def expression(self, node: ast.AST | None, env: dict[str, Value]) -> Value:
        self.steps += 1
        if self.steps > 40_000:
            self.boundary(node, "Module topology statement budget exceeded")
            return Value()
        if isinstance(node, ast.Lambda):
            return self.local_callable(node, env)
        if isinstance(node, ast.GeneratorExp):
            structural = self.affects_modules(node, env) or any(
                self.capability(self.reference(generator.iter, env)) for generator in node.generators)
            return Value(generator=(node, dict(env)), deferred=structural)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            return self.comprehension(node, env)
        if isinstance(node, ast.NamedExpr):
            value = self.expression(node.value, env)
            self.assign(node.target, value, env)
            return value
        if isinstance(node, ast.Name):
            return self.name(node.id, env)
        if isinstance(node, ast.Constant):
            return Value(literal=node.value) if type(node.value) in (int, float, bool, str, type(None)) else Value()
        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            values = [self.expression(child, env) for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)]
            if isinstance(node, ast.JoinedStr) and all(type(value.literal) is str for value in values):
                text = ''.join(value.literal for value in values)
                if len(text) <= 4096:
                    return Value(literal=text)
            if (
                isinstance(node, ast.FormattedValue)
                and values
                and type(values[0].literal) in (int, float, str, bool, type(None))
                and node.format_spec is None
                and len(str(values[0].literal)) <= 1024
            ):
                return Value(literal=repr(values[0].literal) if node.conversion == 114 else str(values[0].literal))
            return Value(origins=dependencies(values).origins, origin_guards=dependencies(values).origin_guards)
        if isinstance(node, ast.Attribute):
            base = self.expression(node.value, env)
            if base.raises:
                return base
            if base.module is not None:
                descriptor = inspect.getattr_static(type(self.module(base.module)), node.attr, None)
                if isinstance(descriptor, property) and descriptor.fget is not None:
                    return self.read_callable(descriptor.fget, [base], {})
            return self.attribute(base, node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            items = tuple(self.expression(item, env) for item in node.elts)
            return Value(items=items, mutable=isinstance(node, ast.List), origins=dependencies(list(items)).origins, origin_guards=dependencies(list(items)).origin_guards)
        if isinstance(node, ast.Dict):
            entries = [(self.expression(key, env).literal, self.expression(value, env))
                       for key, value in zip(node.keys, node.values)]
            if all(type(key) in (str, int) for key, _ in entries):
                return Value(mapping=dict(entries), origins=dependencies([value for _, value in entries]).origins)
            return Value(deferred=any(self.capability(value) for _, value in entries),
                         origins=dependencies([value for _, value in entries]).origins)
        if isinstance(node, ast.Subscript):
            base = self.expression(node.value, env)
            sequence = self.sequence(base)
            index_value = self.expression(node.slice, env)
            index = index_value.literal
            if base.mapping is not None and type(index) in (str, int) and index in base.mapping:
                return base.mapping[index]
            if base.mapping is not None and self.capability(base):
                self.boundary(node, "Dynamic module/callback mapping selection; target not established")
            if sequence is not None and type(index) is int and -len(sequence) <= index < len(sequence):
                return sequence[index]
            if sequence is not None and isinstance(node.slice, ast.Slice):
                bounds = [self.expression(item, env).literal if item else None for item in
                          (node.slice.lower, node.slice.upper, node.slice.step)]
                if all(v is None or type(v) is int for v in bounds) and bounds[2] != 0:
                    return Value(items=sequence[slice(*bounds)])
            if base.module is not None and isinstance(self.module(base.module), torch.nn.ModuleDict) and type(index) is str:
                child = f"{base.module}.{index}".lstrip(".")
                if child in self.raw["modules"]:
                    return Value(module=child)
            if base.module is not None and isinstance(self.module(base.module), torch.nn.ModuleDict) and index is UNKNOWN:
                symbol = f"[selection {self.path}:{ast.unparse(node.slice)}]"
                return Value(choices=tuple(Value(module=f"{base.module}.{name}".lstrip('.'), selection=f"{symbol}={name!r}")
                                           for name in self.module(base.module)._modules))
            if sequence is not None and index_value.index is not None and self.capability(base):
                symbol = f"[selection {index_value.index}]"
                self.boundary(node, "Runtime enumeration selects from a bounded registered container", blocking=False,
                              category="conditional_control_flow", selector=index_value.index,
                              candidates=[item.module for item in sequence], execution_order="unverified")
                return Value(choices=tuple(replace(item, selection=f"{symbol}={position}") for position, item in enumerate(sequence)))
            if sequence is not None and self.capability(base):
                self.boundary(node, "Dynamic registered-module selection; target not established")
            return dependencies([base, self.expression(node.slice, env)])
        if isinstance(node, ast.IfExp):
            test = self.truth(self.expression(node.test, env))
            condition = ast.unparse(node.test)[:240]
            self.current_report["conditions"].append({"condition": condition,
                "resolution": test if type(test) is bool else "runtime_conditional", **self.evidence(node)})
            if type(test) is bool:
                return self.expression(node.body if test else node.orelse, env)
            self.conditions.append(condition)
            left = self.expression(node.body, env)
            self.conditions.pop()
            self.conditions.append(f"not ({condition})")
            right = self.expression(node.orelse, env)
            self.conditions.pop()
            return merge(left, right)
        if isinstance(node, ast.BoolOp):
            result = self.expression(node.values[0], env)
            depth = len(self.conditions)
            for index, operand in enumerate(node.values[1:]):
                known = self.truth(result)
                if type(known) is bool and ((isinstance(node.op, ast.And) and not known)
                                           or (isinstance(node.op, ast.Or) and known)):
                    break
                if type(known) is not bool:
                    self.conditions.append(f"short-circuit operand {index + 1}: {ast.unparse(node)[:240]}")
                next_value = self.expression(operand, env)
                decisive = (self.truth(next_value) is False and isinstance(node.op, ast.And)
                            or self.truth(next_value) is True and isinstance(node.op, ast.Or))
                result = next_value if type(known) is bool or decisive else merge(result, next_value)
            del self.conditions[depth:]
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.truth(self.expression(node.operand, env))
            return Value(literal=not value) if type(value) is bool else Value()
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand, env)
            return Value(literal=-value.literal if isinstance(node.op, ast.USub) else value.literal) if type(value.literal) is int else dependencies([value])
        if isinstance(node, ast.BinOp):
            left, right = self.expression(node.left, env), self.expression(node.right, env)
            if isinstance(node.op, ast.BitOr) and type(left.type_object) is type and type(right.type_object) is type:
                return Value(type_object=left.type_object | right.type_object)
            if (
                isinstance(node.op, ast.Add)
                and left.items is not None
                and right.items is not None
                and len(left.items) + len(right.items) <= limits.MAX_MODEL_LAYERS
            ):
                return Value(items=left.items + right.items)
            arithmetic = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Div: operator.truediv,
                          ast.BitAnd: operator.and_, ast.BitOr: operator.or_}
            operation = arithmetic.get(type(node.op))
            if operation and type(left.literal) in (int, float) and type(right.literal) in (int, float) and max(abs(left.literal), abs(right.literal)) <= 1_000_000:
                try:
                    value = operation(left.literal, right.literal)
                    if abs(value) <= 1_000_000:
                        return Value(literal=value)
                except (TypeError, ZeroDivisionError, OverflowError):
                    pass
            return dependencies([left, right])
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.expression(node.left, env), self.expression(node.comparators[0], env)
            comparison = node.ops[0]
            if isinstance(comparison, (ast.Is, ast.IsNot)):
                if left.module is not None and right.literal is None or right.module is not None and left.literal is None:
                    return Value(literal=isinstance(comparison, ast.IsNot))
                if left.module is not None and right.module is not None and not (left.method or right.method):
                    equal = self.module(left.module) is self.module(right.module)
                    return Value(literal=not equal if isinstance(comparison, ast.IsNot) else equal)
            if type(left.literal) in (str, int) and right.mapping is not None and isinstance(comparison, (ast.In, ast.NotIn)):
                if left.literal in right.optional_keys:
                    return Value()
                present = left.literal in right.mapping
                return Value(literal=not present if isinstance(comparison, ast.NotIn) else present)
            if type(left.literal) in (str, int) and right.items is not None and all(item.literal is not UNKNOWN for item in right.items) and isinstance(comparison, (ast.In, ast.NotIn)):
                present = any(item.literal == left.literal for item in right.items)
                return Value(literal=not present if isinstance(comparison, ast.NotIn) else present)
            if left.literal is not UNKNOWN and right.literal is not UNKNOWN:
                comparison = node.ops[0]
                if isinstance(comparison, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                    equal = left.literal == right.literal
                    return Value(literal=not equal if isinstance(comparison, (ast.NotEq, ast.IsNot)) else equal)
                comparisons = {ast.Lt: operator.lt, ast.LtE: operator.le,
                               ast.Gt: operator.gt, ast.GtE: operator.ge}
                compare = comparisons.get(type(comparison))
                if compare is not None and type(left.literal) is type(right.literal) and type(left.literal) in (int, str):
                    return Value(literal=compare(left.literal, right.literal))
            return dependencies([left, right])
        if isinstance(node, ast.Call):
            target = self.expression(node.func, env)
            if target.raises:
                return target
            args, unknown_star, unknown_positional = [], False, False
            for argument in node.args:
                value = self.expression(argument.value if isinstance(argument, ast.Starred) else argument, env)
                if value.raises:
                    return value
                if isinstance(argument, ast.Starred):
                    if value.items is None:
                        unknown_star = unknown_positional = True
                        args.append(dependencies([value]))
                    else:
                        args.extend(value.items)
                else:
                    args.append(value)
            kwargs = {}
            for keyword in node.keywords:
                value = self.expression(keyword.value, env)
                if value.raises:
                    return value
                if keyword.arg is not None:
                    kwargs[keyword.arg] = value
                elif value.mapping is not None and all(type(key) is str for key in value.mapping):
                    kwargs.update(value.mapping)
                else:
                    unknown_star = True
                    kwargs[None] = value
            if any(target.intrinsic is function for function in (math.log2, math.ceil, math.floor, math.sqrt)) and args:
                if all(type(value.literal) in (int, float) and abs(value.literal) <= 1_000_000 for value in args):
                    try:
                        result = target.intrinsic(*(value.literal for value in args))
                        if math.isfinite(result) and abs(result) <= 1_000_000:
                            return Value(literal=result)
                    except (TypeError, ValueError, OverflowError):
                        pass
                return dependencies(args)
            if target.script_proof is not None and not any(self.capability(value) for value in [*args, *kwargs.values()]):
                self.boundary(node, "Native scripted tensor computation has no module calls", blocking=False, **target.script_proof)
                return dependencies([*args, *kwargs.values()])
            if target.intrinsic is torch.nn.Parameter:
                self.boundary(node, "Source constructs unregistered trainable state; checkpoint architecture is incomplete")
                return Value(deferred=True)
            if target.virtual is not None:
                implementation, fields = target.virtual
                if target.factory:
                    if any(self.capability(value) and value.virtual is None for value in [*args, *kwargs.values()]):
                        self.boundary(node, "Unregistered helper constructor receives unresolved model capabilities")
                        return Value(deferred=True)
                    instance = Value(virtual=(implementation, {}))
                    self.read_callable(implementation.__init__, [instance, *args], kwargs, unknown_star=unknown_star)
                    self.boundary(node, "Unregistered helper constructor inspected without execution", blocking=False,
                                  implementation=f"{implementation.__module__}.{implementation.__qualname__}",
                                  source_sha256=self.callable_hash(implementation.__init__))
                    return instance
                method = target.method or "forward"
                function = inspect.getattr_static(implementation, method, None)
                if method in {"to", "cpu", "cuda", "float", "double", "half", "eval", "train"} and function is inspect.getattr_static(torch.nn.Module, method):
                    return replace(target, method=None)
                if method == "register_buffer" and function is torch.nn.Module.register_buffer:
                    if args and type(args[0].literal) is str:
                        fields[args[0].literal] = args[1] if len(args) > 1 else Value()
                    return Value(literal=None)
                if method == "register_parameter" and function is torch.nn.Module.register_parameter:
                    self.boundary(node, "Unregistered helper registers trainable state outside checkpoint ownership")
                    return Value(deferred=True)
                return self.read_callable(function, [replace(target, method=None), *args], kwargs, unknown_star=unknown_star)
            if target.choices is not None:
                values = []
                for choice in target.choices:
                    if choice.selection:
                        self.conditions.append(choice.selection)
                    if choice.module is not None and not choice.method:
                        values.append(self.module_call(choice, node, args, kwargs))
                    elif choice.raises:
                        pass
                    else:
                        self.boundary(node, "A finite callable alternative has unresolved module effects")
                    if choice.selection:
                        self.conditions.pop()
                return dependencies(values)
            if target.primitive is not None and not any(self.capability(value) for value in [*args, *kwargs.values()]):
                self.boundary(node, "Unregistered native basic-module expression retained as computation detail",
                              blocking=False, **primitive_proof(target.primitive))
                return Value(primitive=target.primitive) if target.factory else dependencies([*args, *kwargs.values()])
            if target.local_function is not None:
                tree, closure, location, globals_snapshot = target.local_function
                local = dict(closure)
                if unknown_positional and (any(self.capability(value) for value in [*args, *kwargs.values()]) or self.affects_modules(tree, local)):
                    self.boundary(node, "Unknown positional star arguments may change local callback binding")
                    return dependencies([*args, *kwargs.values()])
                parameters = [*tree.args.posonlyargs, *tree.args.args]
                for index, parameter in enumerate(parameters):
                    local[parameter.arg] = args[index] if index < len(args) else kwargs.get(parameter.arg,
                        (target.local_defaults or {}).get(parameter.arg, Value(deferred=unknown_star)))
                for parameter in tree.args.kwonlyargs:
                    local[parameter.arg] = kwargs.get(parameter.arg, (target.local_defaults or {}).get(parameter.arg, Value(deferred=unknown_star)))
                if tree.args.vararg:
                    local[tree.args.vararg.arg] = Value(items=tuple(args[len(parameters):]) if not unknown_positional else None, deferred=unknown_positional)
                if tree.args.kwarg:
                    local[tree.args.kwarg.arg] = Value(mapping={name: value for name, value in kwargs.items() if name is not None and name not in {parameter.arg for parameter in [*parameters, *tree.args.kwonlyargs]}})
                old_location, old_globals = self.location, self.globals
                self.location, self.globals = location, globals_snapshot
                key = ('local', id(tree))
                if key in self.active or len(self.active) >= 8:
                    self.boundary(node, "Recursive local callback; module effects not established")
                    self.location, self.globals = old_location, old_globals
                    return dependencies([*args, *kwargs.values()])
                self.active.add(key)
                try:
                    if isinstance(tree, ast.Lambda):
                        return self.expression(tree.body, local)
                    result, flow = self.statements(tree.body, local)
                    return replace(result, raises=True) if flow == 'raise' else result
                finally:
                    self.active.remove(key)
                    self.location, self.globals = old_location, old_globals
            if target.intrinsic is itertools.groupby and len(args) > 1:
                count = sum(len(calls) for calls in self.calls.values())
                probe = ast.Call(func=ast.Name(id='_key_callback', ctx=ast.Load()), args=[ast.Name(id='_item', ctx=ast.Load())], keywords=[])
                self.expression(probe, {'_key_callback': args[1], '_item': Value()})
                if sum(len(calls) for calls in self.calls.values()) != count:
                    self.boundary(node, "Deferred iterator callback has module calls; consumption not established")
                return replace(dependencies([args[0]]), element=Value(items=(Value(), Value())))
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"items", "keys", "values"}:
                receiver = self.reference(node.func.value, env)
                if receiver.mapping is not None:
                    if node.func.attr == "items":
                        return Value(items=tuple(Value(items=(Value(literal=key), value)) for key, value in receiver.mapping.items()))
                    if node.func.attr == "keys":
                        return Value(items=tuple(Value(literal=key) for key in receiver.mapping))
                    return Value(items=tuple(receiver.mapping.values()))
            if isinstance(node.func, ast.Attribute) and node.func.attr in {'get', 'pop', 'index'}:
                receiver = self.reference(node.func.value, env)
                key = args[0].literal if args else UNKNOWN
                if node.func.attr in {'get', 'pop'} and receiver.mapping is not None:
                    default = args[1] if len(args) > 1 else Value(literal=None)
                    if not receiver.mapping or type(key) in (str, int):
                        result = receiver.mapping.get(key, default)
                        if key in receiver.optional_keys:
                            result = merge(result, default)
                        if node.func.attr == 'pop' and isinstance(node.func.value, ast.Name):
                            env[node.func.value.id] = replace(receiver, mapping={name: value for name, value in receiver.mapping.items() if name != key}, optional_keys=receiver.optional_keys - {key})
                        return result
                if node.func.attr == 'index' and receiver.items is not None and type(key) in (str, int):
                    for index, item in enumerate(receiver.items):
                        if item.literal == key:
                            return Value(literal=index)
            if target.namespace is not None and target.method == "get_interface":
                if unknown_positional or unknown_star:
                    self.boundary(node, "Registry selector binding is unknown because call arguments are expanded")
                    return Value(deferred=True)
                selector_name = "experts_implementation" if type(target.namespace) is ExpertsInterface else "attn_implementation"
                selector = args[0].literal if args else kwargs.get(selector_name, Value()).literal
                default = args[1].function if len(args) > 1 else kwargs.get("default", Value()).function
                selected, proof = attention_dispatch(target.namespace, selector, default)
                if selected is not MISSING and inspect.isfunction(selected):
                    self.current_report.setdefault("dispatches", []).append({**self.evidence(node), **proof,
                        "selected_callable": f"{selected.__module__}.{selected.__qualname__}"})
                    return Value(function=selected)
                self.boundary(node, "Callable dispatch target not established from native metadata")
                return Value(deferred=True)
            if unknown_positional and (target.module is not None and target.method or
                                       target.function is not None and (any(self.capability(v) for v in [*args, *kwargs.values()])
                                       or self.function_captures_modules(target.function))):
                self.boundary(node, "Unknown positional star arguments may change module/callback binding")
                return dependencies([*args, *kwargs.values()])
            if target.module is not None:
                if target.method:
                    return self.method(target.module, target.method, args, kwargs, unknown_star=unknown_star)
                return self.module_call(target, node, args, kwargs)
            if target.function is inspect.signature:
                return Value()
            if target.function is not None and (target.bound_self is not None or any(self.capability(v) for v in [*args, *kwargs.values()])
                                                or self.function_captures_modules(target.function)):
                before_calls = sum(len(calls) for calls in self.calls.values())
                before_gaps = sum(item.get("blocking", True) for item in self.current_report["boundaries"])
                result = self.read_callable(target.function, [target.bound_self, *args] if target.bound_self is not None else args,
                                            kwargs, unknown_star=unknown_star)
                new_gaps = sum(item.get("blocking", True) for item in self.current_report["boundaries"]) - before_gaps
                if not new_gaps:
                    calls = sum(len(calls) for calls in self.calls.values()) - before_calls
                    self.boundary(node, "Callable module effects inspected without tensor execution", blocking=False,
                        category="resolved_module_calls" if calls or self.capability(result) else "computation_detail",
                        callable=f"{target.function.__module__}.{target.function.__qualname__}",
                        source_sha256=self.callable_hash(target.function), module_calls=calls, helper_executed=False)
                return result
            if (isinstance(node.func, ast.Name) and node.func.id not in env
                    and getattr(builtins, node.func.id, None) is not None
                    and self.globals.get(node.func.id, getattr(builtins, node.func.id)) is getattr(builtins, node.func.id)):
                name = node.func.id
                intrinsic = self.globals.get(name, getattr(builtins, name, None))
                sequence = (self.comprehension(args[0].generator[0], dict(args[0].generator[1]), first_only=True).items
                            if name == "next" and args and args[0].generator is not None
                            else self.sequence(args[0]) if args else None)
                if intrinsic is isinstance and len(args) == 2:
                    types = args[1].items if args[1].items is not None else (args[1],)
                    classes = tuple(implementation for value in types for implementation in
                                    (value.type_object.__args__ if type(value.type_object) is UnionType else (value.type_object,)))
                    if classes and all(type(implementation) is type for implementation in classes):
                        source_type = (type(args[0].literal) if args[0].literal is not UNKNOWN else
                                       dict if args[0].mapping is not None else
                                       list if args[0].items is not None and args[0].mutable else
                                       tuple if args[0].items is not None else
                                       type(self.module(args[0].module)) if args[0].module is not None else None)
                        if source_type is not None:
                            return Value(literal=issubclass(source_type, classes))
                if any(intrinsic is function for function in (isinstance, issubclass, type, id)):
                    return Value()
                if name == "getattr" and len(args) in (2, 3) and args[0].module is not None and type(args[1].literal) is str:
                    owner = self.module(args[0].module)
                    attribute = args[1].literal
                    present = (attribute in vars(owner) or attribute in owner._modules or attribute in owner._parameters
                               or attribute in owner._buffers or inspect.getattr_static(type(owner), attribute, MISSING) is not MISSING)
                    return self.attribute(args[0], attribute) if present else args[2] if len(args) == 3 else Value(deferred=True)
                if name in {'getattr', 'hasattr'} and len(args) in (2, 3) and args[0].config is not None and type(args[1].literal) is str:
                    value = static_attribute(args[0].config, args[1].literal)
                    return Value(literal=value is not MISSING) if name == 'hasattr' else self.value(value) if value is not MISSING else args[2] if len(args) == 3 else Value(raises=True)
                if name in {"getattr", "hasattr"} and len(args) in (2, 3) and args[0].virtual is not None and type(args[1].literal) is str:
                    implementation, fields = args[0].virtual
                    attribute = args[1].literal
                    present = attribute in fields or inspect.getattr_static(implementation, attribute, MISSING) is not MISSING
                    if attribute in {"__name__", "__qualname__", "__module__"}:
                        present = args[0].factory
                    return Value(literal=present) if name == 'hasattr' else self.attribute(args[0], attribute) if present else args[2] if len(args) == 3 else Value(raises=True)
                if name == "hasattr" and len(args) == 2 and args[0].module is not None and type(args[1].literal) is str:
                    owner = self.module(args[0].module)
                    attribute = args[1].literal
                    present = (attribute in vars(owner) or attribute in owner._modules or attribute in owner._parameters
                               or attribute in owner._buffers or inspect.getattr_static(type(owner), attribute, UNKNOWN) is not UNKNOWN)
                    return Value(literal=present)
                if name in {"set", "frozenset"} and sequence is not None and all(
                        type(value.literal) in (str, int, bool, type(None)) for value in sequence):
                    members = {value.literal for value in sequence}
                    return Value(items=tuple(Value(literal=value) for value in sorted(members, key=repr)), unordered=True)
                if name == "len" and sequence is not None:
                    return Value() if args[0].optional_keys else Value(literal=len(sequence))
                if name == 'str' and args and isinstance(args[0].namespace, type) and type(args[0].namespace) is type:
                    return Value(literal=str(args[0].namespace))
                if name == "next" and sequence is not None:
                    return sequence[0] if sequence else args[1] if len(args) > 1 else Value()
                if name in {'int', 'float'} and args and type(args[0].literal) in (int, float) and abs(args[0].literal) <= 1_000_000:
                    return Value(literal=intrinsic(args[0].literal))
                if name in {"list", "tuple"} and sequence is not None:
                    return Value(items=sequence, mutable=name == 'list')
                if name == "enumerate" and sequence is not None:
                    return Value(items=tuple(Value(items=(Value(literal=i), item)) for i, item in enumerate(sequence)))
                if name == "enumerate" and args:
                    symbol = f"{self.location.get('method')}:{self.evidence(node)['line']}:{ast.unparse(node)}"
                    return Value(element=Value(items=(Value(index=symbol), args[0].element or Value())))
                if name == "range" and args and all(type(v.literal) is int for v in args):
                    try:
                        values = range(*(v.literal for v in args))
                        if len(values) <= limits.MAX_MODEL_LAYERS:
                            return Value(items=tuple(Value(literal=i) for i in values))
                    except (ValueError, TypeError, OverflowError):
                        pass
                if name == "zip" and args:
                    sequences = [self.sequence(value) for value in args]
                    bounded = [items for items in sequences if items is not None]
                    if bounded:
                        length = min(map(len, bounded))
                        return Value(items=tuple(Value(items=tuple(items[index] if items is not None else value.element or Value()
                                                                  for items, value in zip(sequences, args))) for index in range(length)),
                                     conditional_items=len(bounded) != len(sequences))
                    return Value(element=Value(items=tuple(value.element or Value() for value in args)))
            if target.deferred or any(self.capability(value) for value in [*args, *kwargs.values()]):
                self.boundary(node, "Opaque callable receives a module or module callback; invocation not established")
            elif (isinstance(node.func, ast.Name) and node.func.id in env and target.literal is UNKNOWN
                  and target.function is None and target.intrinsic is None and target.primitive is None and target.script_proof is None):
                self.boundary(node, "Runtime callable target not established from source or native metadata")
            # Functional computations are outside the architecture graph. Carry
            # dependencies through them without claiming an identity transform.
            receiver = Value(origins=target.origins, origin_guards=target.origin_guards, deferred=target.deferred)
            return dependencies([receiver, *args, *kwargs.values()])
        return dependencies([self.expression(child, env) for child in ast.iter_child_nodes(node)
                             if isinstance(child, ast.expr)]) if node is not None else Value()

    def module_call(self, target: Value, node: ast.AST, args: list[Value], kwargs: dict[str, Value]) -> Value:
        proof = self.evidence(node)
        self.calls.setdefault(target.module, []).append(proof)
        inputs = dependencies([*args, *kwargs.values()])
        selections = {item.split(']=', 1)[0]: item for item in self.conditions if item.startswith('[selection ') and ']=' in item}
        for source in sorted(inputs.origins):
            if source == target.module:
                continue
            guards = [list(conditions) for origin, conditions in inputs.origin_guards if origin == source and conditions]
            viable = [guard for guard in guards if not any(
                condition.startswith('[selection ') and ']=' in condition and condition.split(']=', 1)[0] in selections
                and selections[condition.split(']=', 1)[0]] != condition for condition in guard)]
            if guards and not viable:
                continue
            self.edges.append({"source": source, "target": target.module, **proof, "source_conditions": viable})
        return Value(origins=frozenset({target.module}), origin_guards=((target.module, tuple(self.conditions)),))

    def assign(self, target: ast.AST, value: Value, env: dict[str, Value]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, ast.Attribute) and (base := self.reference(target.value, env)).virtual is not None:
            base.virtual[1][target.attr] = value
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            base = env.get(target.value.id, Value())
            key = self.expression(target.slice, env).literal
            if base.mapping is not None and type(key) in (str, int):
                env[target.value.id] = replace(base, mapping={**base.mapping, key: value}, optional_keys=base.optional_keys - {key})
            elif base.items is not None and base.mutable and type(key) is int and -len(base.items) <= key < len(base.items):
                items = list(base.items)
                items[key] = value
                env[target.value.id] = replace(base, items=tuple(items))
            else:
                self.boundary(target, "Mapping assignment target not established", blocking=self.capability(base) or self.capability(value))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                self.assign(item, value.items[index] if value.items and index < len(value.items) else value, env)

    def has_control_transfer(self, nodes: list[ast.stmt], env: dict[str, Value]) -> bool:
        """Ignore unreachable break/continue paths only when metadata proves the guard."""
        for node in nodes:
            if isinstance(node, (ast.Break, ast.Continue, ast.Return)):
                return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.If) and not any(isinstance(item, ast.Call) for item in ast.walk(node.test)):
                test = self.truth(self.expression(node.test, env))
                bodies = node.body if test is True else node.orelse if test is False else [*node.body, *node.orelse]
                if self.has_control_transfer(bodies, env):
                    return True
            elif self.has_control_transfer([item for item in ast.iter_child_nodes(node) if isinstance(item, ast.stmt)], env):
                return True
        return False

    def statements(self, statements: list[ast.stmt], env: dict[str, Value]) -> tuple[Value, str | None]:
        depth = len(self.conditions)
        try:
            return self._statements(statements, env)
        finally:
            del self.conditions[depth:]

    def _statements(self, statements: list[ast.stmt], env: dict[str, Value]) -> tuple[Value, str | None]:
        result = Value()
        for node in statements:
            if self.steps > 40_000:
                break
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.expression(node.value, env)
                if value.raises:
                    return value, "raise"
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and self.reference(target.value, env).registry:
                        self.boundary(node, "Registered module or state registry mutation is not reconstructed")
                        continue
                    if isinstance(target, ast.Attribute) and self.reference(target.value, env).virtual is not None:
                        continue
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and (
                            local := env.get(target.value.id)) is not None and not local.registry and (local.mapping is not None or local.mutable):
                        continue
                    if isinstance(target, (ast.Attribute, ast.Subscript)) and (
                            self.capability(self.reference(target, env)) or self.capability(value)):
                        self.boundary(node, "Module registry or module reference mutation is not reconstructed")
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AugAssign):
                reference = self.reference(node.target, env)
                if self.capability(reference) or reference.registry:
                    self.boundary(node, "Augmented module or registry mutation is not reconstructed")
                self.assign(node.target, dependencies([self.expression(node.target, env), self.expression(node.value, env)]), env)
            elif isinstance(node, ast.Expr):
                value = self.expression(node.value, env)
                if value.raises:
                    return value, "raise"
            elif isinstance(node, ast.Return):
                value = self.expression(node.value, env)
                return value, "raise" if value.raises else "return"
            elif isinstance(node, ast.If):
                test = self.truth(self.expression(node.test, env))
                condition = ast.unparse(node.test)[:240]
                self.current_report["conditions"].append({"condition": condition,
                    "resolution": test if type(test) is bool else "runtime_conditional", **self.evidence(node)})
                if type(test) is bool:
                    result, returned = self.statements(node.body if test else node.orelse, env)
                    if returned:
                        return result, True
                else:
                    before = dict(env)
                    branches, returns = [], []
                    for body, label in ((node.body, condition), (node.orelse, f"not ({condition})")):
                        branch = dict(before)
                        self.conditions.append(label)
                        returned_value, returned = self.statements(body, branch)
                        self.conditions.pop()
                        branches.append(branch)
                        returns.append((returned_value, returned))
                    result = merge(returns[0][0], returns[1][0])
                    if all(item[1] for item in returns):
                        return result, returns[0][1] if returns[0][1] == returns[1][1] else "exit"
                    if any(item[1] for item in returns):
                        continuing = 1 if returns[0][1] else 0
                        env.clear()
                        env.update(branches[continuing])
                        self.conditions.append(f"not ({condition})" if continuing else condition)
                    else:
                        for key in set().union(*branches):
                            env[key] = merge(branches[0].get(key, Value()), branches[1].get(key, Value()))
            elif isinstance(node, ast.For):
                iterable = self.expression(node.iter, env)
                sequence = self.sequence(iterable)
                interrupted = self.has_control_transfer(node.body, env)
                if sequence is None or len(sequence) > limits.MAX_MODEL_LAYERS or interrupted:
                    structural = self.affects_modules(node, env) or self.capability(self.reference(node.iter, env))
                    self.boundary(node, "Runtime loop retained as conditional module relationships", blocking=False,
                                  category="conditional_control_flow", iteration_count="unverified")
                    self.conditions.append("unresolved loop")
                    before_loop = dict(env)
                    self.assign(node.target, iterable.element or Value(), env)
                    self.loop_effects.append(structural)
                    try:
                        self.statements(node.body, env)
                    finally:
                        self.loop_effects.pop()
                        self.conditions.pop()
                    for name in set(before_loop) | set(env):
                        env[name] = merge(before_loop.get(name, Value()), env.get(name, Value()))
                elif iterable.unordered:
                    # Each member starts from the same data dependencies. Never
                    # invent a layer chain from a deterministic display order.
                    before, branches = dict(env), []
                    assignments = [item for item in ast.walk(node) if isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign))]
                    targets = [target for item in assignments for target in (item.targets if isinstance(item, ast.Assign) else [item.target])]
                    mutated = {target.value.id for target in targets if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)}
                    target_nodes = {id(item) for target in targets for item in ast.walk(target)}
                    carried = any(not isinstance(target, ast.Subscript) for target in targets) or any(
                        isinstance(item, ast.Name) and item.id in mutated and id(item) not in target_nodes
                        for statement in node.body for item in ast.walk(statement))
                    if carried and self.affects_modules(node, env):
                        self.boundary(node, "Unordered iteration may carry module dependencies between members")
                    for value in sequence:
                        branch = dict(before)
                        self.assign(node.target, value, branch)
                        self.conditions.append(f"unordered metadata member: {value.literal!r}; execution order unverified")
                        self.statements(node.body, branch)
                        self.conditions.pop()
                        branches.append(branch)
                    for branch in branches:
                        for name, value in branch.items():
                            previous = env.get(name, Value())
                            if name in mutated and previous.mapping is not None and value.mapping is not None:
                                env[name] = replace(previous, mapping={**previous.mapping, **value.mapping})
                            elif name != getattr(node.target, "id", None):
                                env[name] = merge(previous, value)
                else:
                    for index, value in enumerate(sequence):
                        if iterable.conditional_items:
                            self.conditions.append(f"runtime zip operand has element {index}")
                        self.assign(node.target, value, env)
                        result, flow = self.statements(node.body, env)
                        if iterable.conditional_items:
                            self.conditions.pop()
                        if flow == "break":
                            break
                        if flow in {"return", "raise", "exit"}:
                            return result, flow
                if node.orelse:
                    self.conditions.append("loop completed without break")
                    self.statements(node.orelse, env)
                    self.conditions.pop()
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    self.expression(item.context_expr, env)
                result, returned = self.statements(node.body, env)
                if returned:
                    return result, True
            elif isinstance(node, (ast.Try, ast.TryStar)):
                self.boundary(node, "Exception control flow retained as conditional module relationships",
                              blocking=False, category="conditional_control_flow")
                before, alternatives, outcomes = dict(env), [], []
                branches = [(node.body, "try body completed without exception"),
                            *[(handler.body, f"exception handler: {ast.unparse(handler.type) if handler.type else 'BaseException'}") for handler in node.handlers]]
                for index, (body, condition) in enumerate(branches):
                    branch = dict(before)
                    self.conditions.append(condition)
                    value, flow = self.statements(body, branch)
                    if index == 0 and not flow:
                        value, flow = self.statements(node.orelse, branch)
                    if node.finalbody:
                        final_value, final_flow = self.statements(node.finalbody, branch)
                        if final_flow:
                            value, flow = final_value, final_flow
                    self.conditions.pop()
                    alternatives.append(branch)
                    outcomes.append((value, flow))
                for branch in alternatives:
                    for key, value in branch.items():
                        env[key] = merge(env.get(key, Value()), value)
                if outcomes and all(flow for _, flow in outcomes):
                    return outcomes[0] if all(flow == outcomes[0][1] for _, flow in outcomes) else (Value(), "exit")
                self.conditions.append("exception control flow continued")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.decorator_list:
                    self.boundary(node, "Local callback decorators require an established callable-effects contract")
                env[node.name] = self.local_callable(node, env)
            elif isinstance(node, (ast.Break, ast.Continue)):
                if self.loop_effects:
                    self.boundary(node, "Loop transfer retained under its source condition", blocking=False,
                                  category="conditional_control_flow")
                else:
                    self.boundary(node, "Loop transfer outside an established loop")
                return Value(), "break" if isinstance(node, ast.Break) else "continue"
            elif isinstance(node, (ast.While, ast.Match)):
                self.boundary(node, "Unsupported module control flow; no execution order asserted",
                              blocking=self.affects_modules(node, env))
            elif isinstance(node, ast.Delete):
                if any(self.capability(self.reference(target, env)) or self.reference(target, env).registry for target in node.targets):
                    self.boundary(node, "Registered module deletion changes architecture")
            elif isinstance(node, ast.Assert):
                self.expression(node.test, env)
            elif isinstance(node, ast.Raise):
                self.expression(node.exc, env)
                return Value(raises=True), "raise"
            elif not isinstance(node, (ast.Pass, ast.Import, ast.ImportFrom)):
                self.boundary(node, "Statement outside bounded module topology support", blocking=self.affects_modules(node, env))
        return result, None
