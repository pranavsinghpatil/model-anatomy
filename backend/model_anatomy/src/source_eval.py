"""Bounded interpretation of Python helpers over symbolic tensors, never tensor execution."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import operator
import sys
import textwrap
from dataclasses import dataclass, field
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any

import torch


class SourceBoundary(Exception):
    """A source dependency could not be established without executing publisher code."""


class MissingAttribute(SourceBoundary):
    """Ordinary static attribute absence, distinct from unsupported dynamic lookup."""


@dataclass(frozen=True)
class Symbol:
    expression: str

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
        return {"shape": [render(d) for d in self.shape], "dtype": self.dtype,
                "expression": self.expression}


@dataclass
class Closure:
    tree: ast.FunctionDef
    env: dict[str, Any]
    globals: dict[str, Any]
    source: dict[str, Any]
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Bound:
    owner: Any
    name: str


@dataclass(frozen=True)
class Mapped:
    function: Any
    dimensions: tuple[Any, ...]


@dataclass(frozen=True)
class FloatInfo:
    dtype: str


@dataclass(frozen=True)
class ModuleState:
    attributes: dict[str, Any]
    evidence: dict[str, Any]


def render(value: Any) -> Any:
    return value.expression if isinstance(value, Symbol) else value


def expression(value: Any) -> str:
    if isinstance(value, (Tensor, Symbol)):
        return value.expression
    if type(value) in (type(None), bool, int, float, str):
        return repr(value)
    return type(value).__name__


DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
          torch.float64: "F64", torch.int64: "I64", torch.int32: "I32", torch.bool: "BOOL"}
TORCH_PRIMITIVES = {name: getattr(torch, name) for name in ("arange", "diff", "vmap", "tensor", "finfo", "where", "ones", "zeros")}


def dtype_name(value: Any) -> str:
    if type(value) is torch.dtype and value in DTYPES:
        return DTYPES[value]
    raise SourceBoundary("Tensor dtype is not established")


class SourceEvaluator:
    """Interpret supported AST and identity-checked primitives with explicit proof records."""

    def __init__(self, *, max_steps: int = 4096, max_depth: int = 16, max_paths: int = 16):
        self.max_steps, self.max_depth, self.max_paths = max_steps, max_depth, max_paths
        self.steps = self.depth = self.paths = 0
        self.sources: list[dict[str, Any]] = []
        self.conditions: list[dict[str, Any]] = []
        self.operations: list[dict[str, Any]] = []
        self.current: dict[str, Any] = {}
        self.owned: set[int] = set()
        self.property_getters: set[int] = set()
        self.last_node: ast.AST | None = None
        self.failure: dict[str, Any] | None = None

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps or self.depth > self.max_depth:
            raise SourceBoundary("Static helper analysis exceeded its step or recursion budget")

    def record(self, node: ast.AST, rule: str, **details: Any) -> None:
        self.operations.append({"rule": rule, **self.current,
                                "line": getattr(node, "lineno", 0), **details})

    def evidence(self) -> dict[str, Any]:
        return {"rule": "bounded_python_source", "sources": self.sources,
                "conditions": self.conditions, "operations": self.operations,
                "tensor_execution": False, "forward_executed": False,
                **({"failure": self.failure} if self.failure else {}),
                "scope": "Recorded symbolic inputs and native configuration only"}

    def container(self, value: Any) -> Any:
        self.owned.add(id(value))
        return value

    def source(self, function: FunctionType) -> Closure:
        if type(function) is not FunctionType:
            raise SourceBoundary("Callable is not an inspectable Python function")
        try:
            text = textwrap.dedent(inspect.getsource(function))
            if len(text.encode()) > 256 * 1024:
                raise SourceBoundary("Helper source exceeds its text budget")
            tree = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef))
        except (OSError, TypeError, SyntaxError, StopIteration) as exc:
            raise SourceBoundary("Helper source is unavailable") from exc
        if ((tree.decorator_list and id(function) not in self.property_getters)
                or function.__closure__ or "__wrapped__" in vars(function)):
            raise SourceBoundary("Decorated functions and runtime closure state are unsupported")
        if sum(1 for _ in ast.walk(tree)) > 4096:
            raise SourceBoundary("Helper source exceeds the AST budget")
        source = {"source": f"{function.__module__}.{function.__qualname__}",
                  "source_sha256": hashlib.sha256(text.encode()).hexdigest()}
        if source not in self.sources:
            self.sources.append(source)
        positional = tree.args.posonlyargs + tree.args.args
        defaults = function.__defaults__ or ()
        bound_defaults = dict(zip([p.arg for p in positional][-len(defaults):], defaults)) if defaults else {}
        bound_defaults.update(function.__kwdefaults__ or {})
        return Closure(tree, {}, function.__globals__, source, bound_defaults)

    def bind(self, fn: Closure, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        params = fn.tree.args
        positional = params.posonlyargs + params.args
        if len(args) > len(positional) and params.vararg is None:
            raise SourceBoundary("Too many positional helper arguments")
        names = [p.arg for p in positional]
        bound = dict(zip(names, args))
        kwargs = dict(kwargs)
        for p in positional[len(args):] + params.kwonlyargs:
            if p.arg in kwargs and p not in params.posonlyargs:
                bound[p.arg] = kwargs.pop(p.arg)
        positional_only = {p.arg for p in params.posonlyargs}
        if any(name in kwargs for name in names[:len(args)] if name not in positional_only):
            raise SourceBoundary("Helper argument supplied more than once")
        for p in positional + params.kwonlyargs:
            if p.arg not in bound:
                if p.arg not in fn.defaults:
                    raise SourceBoundary(f"Required helper argument is unresolved: {p.arg}")
                bound[p.arg] = fn.defaults[p.arg]
        if params.vararg:
            bound[params.vararg.arg] = tuple(args[len(positional):])
        if kwargs and params.kwarg is None:
            raise SourceBoundary("Unexpected helper keyword arguments")
        if params.kwarg:
            bound[params.kwarg.arg] = self.container(kwargs)
        return {**fn.env, **bound}

    def invoke(self, function: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        self.tick()
        if isinstance(function, Mapped):
            return self.vectorized(function, args, kwargs, node)
        if isinstance(function, Bound):
            return self.method(function.owner, function.name, args, kwargs, node)
        primitive = self.primitive(function, args, kwargs, node)
        if primitive is not NotImplemented:
            return primitive
        fn = function if isinstance(function, Closure) else self.source(function)
        previous = self.current
        self.current = fn.source
        self.depth += 1
        try:
            env = self.bind(fn, args, kwargs)
            outcomes = self.block(fn.tree.body, env, fn.globals)
            if any(not returned for _, returned, _ in outcomes):
                outcomes = [(value if returned else None, True, state) for value, returned, state in outcomes]
            values = [value for value, _, _ in outcomes]
            result = values[0]
            for value in values[1:]:
                result = self.merge(result, value)
            return result
        except SourceBoundary as exc:
            if self.failure is None:
                self.failure = {**self.current, "line": getattr(self.last_node, "lineno", 0), "reason": str(exc)}
            raise
        finally:
            self.depth -= 1
            self.current = previous

    def merge(self, left: Any, right: Any) -> Any:
        if isinstance(left, Tensor) and isinstance(right, Tensor) and len(left.shape) == len(right.shape) and left.dtype == right.dtype:
            shape = tuple(a if a == b else Symbol(f"choice({expression(a)}, {expression(b)})") for a, b in zip(left.shape, right.shape))
            return Tensor(shape, left.dtype, f"choice({left.expression}, {right.expression})")
        if type(left) is tuple and type(right) is tuple and len(left) == len(right):
            return tuple(self.merge(a, b) for a, b in zip(left, right))
        if ((type(left) in (type(None), bool, int, str) and type(right) is type(left) and left == right)
                or isinstance(left, Symbol) and isinstance(right, Symbol) and left == right):
            return left
        raise SourceBoundary("Symbolic branches have incompatible helper return types")

    def attribute(self, owner: Any, name: str) -> Any:
        if isinstance(owner, ModuleState):
            if name not in owner.attributes:
                raise SourceBoundary(f"Module attribute has no static evidence: {name}")
            self.record(ast.Constant(value=None), "native_module_attribute", name=name, evidence=owner.evidence)
            return owner.attributes[name]
        if isinstance(owner, FloatInfo) and name == "min":
            return Symbol(f"finfo({owner.dtype}).min")
        if isinstance(owner, Tensor):
            if name == "shape":
                return owner.shape
            if name == "ndim":
                return len(owner.shape)
            if name == "dtype":
                return next(k for k, v in DTYPES.items() if v == owner.dtype)
            if name == "device":
                return Symbol("input_device")
            if name in {"to", "expand", "cumsum", "new_ones", "size", "dim", "view", "reshape"}:
                return Bound(owner, name)
            raise SourceBoundary(f"Unsupported symbolic tensor attribute: {name}")
        if type(owner) in (list, dict) and name in {"extend", "pop"}:
            return Bound(owner, name)
        if type(owner) is ModuleType:
            if name not in vars(owner):
                raise SourceBoundary(f"Module attribute is unavailable: {name}")
            return vars(owner)[name]
        name = self.attribute_name(owner, name)
        try:
            value = inspect.getattr_static(owner, name)
        except AttributeError as exc:
            if inspect.getattr_static(type(owner), "__getattr__", None) is not None:
                raise SourceBoundary(f"Dynamic attribute fallback is unsupported: {name}") from exc
            raise MissingAttribute(f"Attribute is unavailable: {name}") from exc
        if type(value) is property:
            self.property_getters.add(id(value.fget))
            return self.invoke(value.fget, [owner], {}, ast.Constant(value=None))
        if type(value) is FunctionType:
            return Bound(owner, name)
        if inspect.getattr_static(type(value), "__get__", None) is not None:
            raise SourceBoundary(f"Dynamic attribute descriptor is unsupported: {name}")
        if type(value) in (type(None), bool, int, float, str):
            self.record(ast.Constant(value=None), "static_attribute", name=name, value=value)
        return value

    def attribute_name(self, owner: Any, name: str) -> str:
        """Accept ordinary lookup or the source-verified native config alias contract."""
        implementation = inspect.getattr_static(type(owner), "__getattribute__", object.__getattribute__)
        if implementation is object.__getattribute__:
            return name
        from transformers import PretrainedConfig

        if implementation is not PretrainedConfig.__getattribute__:
            raise SourceBoundary("Custom attribute lookup requires source proof")
        text = textwrap.dedent(inspect.getsource(implementation))
        expected = '''def __getattribute__(self, key):
    if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
        key = super().__getattribute__("attribute_map")[key]
    return super().__getattribute__(key)
'''
        if ast.dump(ast.parse(text), include_attributes=False) != ast.dump(ast.parse(expected), include_attributes=False):
            raise SourceBoundary("Native config attribute lookup contract has changed")
        mapping = inspect.getattr_static(owner, "attribute_map", {})
        if type(mapping) is not dict or len(mapping) > 128 or any(type(k) is not str or type(v) is not str for k, v in mapping.items()):
            raise SourceBoundary("Native config attribute map is not a bounded string mapping")
        resolved = mapping.get(name, name) if name != "attribute_map" else name
        self.record(ast.Constant(value=None), "native_config_attribute_alias", name=name, resolved_name=resolved,
                    lookup_source="transformers.PretrainedConfig.__getattribute__",
                    lookup_source_sha256=hashlib.sha256(text.encode()).hexdigest())
        return resolved

    def assign(self, target: ast.AST, value: Any, env: dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and type(value) in (tuple, list) and len(value) == len(target.elts):
            for part, item in zip(target.elts, value):
                self.assign(part, item, env)
        else:
            raise SourceBoundary("Helper assignment effects are unsupported")

    def block(self, nodes: list[ast.stmt], env: dict[str, Any], globals: dict[str, Any]) -> list[tuple[Any, bool, dict]]:
        for index, node in enumerate(nodes):
            self.last_node = node
            self.tick()
            if isinstance(node, ast.Return):
                return [(self.eval(node.value, env, globals) if node.value else None, True, env)]
            if isinstance(node, ast.Assign):
                value = self.eval(node.value, env, globals)
                for target in node.targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self.assign(node.target, self.eval(node.value, env, globals), env)
            elif isinstance(node, ast.AugAssign):
                self.assign(node.target, self.binary(self.eval(node.target, env, globals), self.eval(node.value, env, globals), node.op), env)
            elif isinstance(node, ast.Expr):
                self.eval(node.value, env, globals)
            elif isinstance(node, ast.FunctionDef):
                if node.decorator_list:
                    raise SourceBoundary("Nested decorators are unsupported")
                positional = node.args.posonlyargs + node.args.args
                defaults = dict(zip([p.arg for p in positional][-len(node.args.defaults):],
                                    [self.eval(n, env, globals) for n in node.args.defaults])) if node.args.defaults else {}
                defaults.update({p.arg: self.eval(n, env, globals) for p, n in zip(node.args.kwonlyargs, node.args.kw_defaults) if n is not None})
                env[node.name] = Closure(node, env, globals, self.current, defaults)
            elif isinstance(node, ast.If):
                condition = self.eval(node.test, env, globals)
                known = type(condition) is bool
                self.conditions.append({**self.current, "line": node.lineno,
                                        "condition": ast.unparse(node.test), "resolved": known,
                                        "value": condition if known else None,
                                        "symbolic": expression(condition)})
                if known:
                    outcomes = self.block(node.body if condition else node.orelse, env, globals)
                else:
                    if not isinstance(condition, Symbol):
                        raise SourceBoundary("A tensor-dependent branch is not statically resolved")
                    self.paths += 2
                    if self.paths > self.max_paths:
                        raise SourceBoundary("Symbolic helper branches exceed the path budget")
                    # Mutable state cannot be safely shared across speculative paths.
                    if any(type(value) in (dict, list) or isinstance(value, (Closure, Mapped, Bound)) for value in env.values()):
                        raise SourceBoundary("Symbolic branch contains mutable local state")
                    outcomes = self.block(node.body, dict(env), globals) + self.block(node.orelse, dict(env), globals)
                finished = []
                for value, returned, state in outcomes:
                    finished.extend([(value, True, state)] if returned else self.block(nodes[index + 1:], state, globals))
                return finished
            elif isinstance(node, ast.For):
                items = self.eval(node.iter, env, globals)
                if type(items) not in (list, tuple) or len(items) > 128 or node.orelse:
                    raise SourceBoundary("Helper loop is not a bounded literal sequence")
                for item in items:
                    self.assign(node.target, item, env)
                    outcomes = self.block(node.body, env, globals)
                    if len(outcomes) != 1 or outcomes[0][1]:
                        raise SourceBoundary("Early or conditional loop returns are unsupported")
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars or self.eval(item.context_expr, env, globals) != ("symbolic_index_context",):
                        raise SourceBoundary("Context manager effects are unsupported")
                outcomes = self.block(node.body, env, globals)
                if len(outcomes) != 1:
                    raise SourceBoundary("Conditional context outcomes are unsupported")
                if outcomes[0][1]:
                    return outcomes
            elif isinstance(node, ast.Pass):
                continue
            else:
                raise SourceBoundary(f"Unsupported helper statement: {type(node).__name__}")
        return [(None, False, env)]

    def eval(self, node: ast.AST, env: dict[str, Any], globals: dict[str, Any]) -> Any:
        self.last_node = node
        self.tick()
        if isinstance(node, ast.Constant):
            if (type(node.value) is int and node.value.bit_length() > 64
                    or type(node.value) is str and len(node.value) > 4096):
                raise SourceBoundary("Source literal exceeds its budget")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in globals:
                return globals[node.id]
            if node.id in vars(builtins):
                return vars(builtins)[node.id]
            raise SourceBoundary(f"Unbound helper name: {node.id}")
        if isinstance(node, ast.Attribute):
            return self.attribute(self.eval(node.value, env, globals), node.attr)
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > 128:
                raise SourceBoundary("Source container exceeds its budget")
            values = [self.eval(item, env, globals) for item in node.elts]
            return self.container(values) if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.GeneratorExp) and len(node.generators) == 1:
            generator = node.generators[0]
            items = self.eval(generator.iter, env, globals)
            if generator.ifs or generator.is_async or type(items) not in (tuple, list) or len(items) > 128:
                raise SourceBoundary("Generator is not a bounded literal sequence")
            values = []
            for item in items:
                local = dict(env)
                self.assign(generator.target, item, local)
                values.append(self.eval(node.elt, local, globals))
            return tuple(values)
        if isinstance(node, ast.UnaryOp):
            value = self.eval(node.operand, env, globals)
            if isinstance(node.op, ast.Not) and type(value) is bool:
                return not value
            if isinstance(node.op, ast.USub) and type(value) in (int, float):
                return -value
            raise SourceBoundary("Unsupported helper unary expression")
        if isinstance(node, ast.BinOp):
            return self.binary(self.eval(node.left, env, globals), self.eval(node.right, env, globals), node.op)
        if isinstance(node, ast.BoolOp):
            values = []
            for index, part in enumerate(node.values):
                value = self.eval(part, env, globals)
                if type(value) is bool:
                    if isinstance(node.op, ast.And) and value is False:
                        return False
                    if isinstance(node.op, ast.Or) and value is True:
                        return True
                elif not isinstance(value, Symbol):
                    raise SourceBoundary("Tensor truth values are not evaluated")
                elif index < len(node.values) - 1:
                    raise SourceBoundary("Symbolic short-circuit evaluation requires conditional effects analysis")
                values.append(expression(value))
            if all(value in {"True", "False"} for value in values):
                return isinstance(node.op, ast.And)
            return Symbol((" and " if isinstance(node.op, ast.And) else " or ").join(values))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.eval(node.left, env, globals), self.eval(node.comparators[0], env, globals)
            op = node.ops[0]
            if isinstance(op, (ast.Is, ast.IsNot)):
                result = left is right
                return not result if isinstance(op, ast.IsNot) else result
            if isinstance(op, (ast.In, ast.NotIn)) and type(right) is dict and type(left) in (int, str):
                result = left in right
                return not result if isinstance(op, ast.NotIn) else result
            if type(left) in (type(None), bool, int, float, str) and type(right) in (type(None), bool, int, float, str):
                operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                              ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
                if type(op) in operations:
                    return operations[type(op)](left, right)
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(left, Symbol) and isinstance(right, Symbol) and left == right:
                return isinstance(op, ast.Eq)
            symbols = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
            if type(op) not in symbols:
                raise SourceBoundary("Unsupported helper comparison")
            expr = f"({expression(left)} {symbols[type(op)]} {expression(right)})"
            self.record(node, "symbolic_comparison", expression=expr)
            tensors = [value for value in (left, right) if isinstance(value, Tensor)]
            return Tensor(self.broadcast(tensors), "BOOL", expr) if tensors else Symbol(expr)
        if isinstance(node, ast.IfExp):
            condition = self.eval(node.test, env, globals)
            if type(condition) is not bool:
                raise SourceBoundary("Conditional helper expression is unresolved")
            return self.eval(node.body if condition else node.orelse, env, globals)
        if isinstance(node, ast.Slice):
            return slice(*(self.eval(item, env, globals) if item else None for item in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Subscript):
            owner, key = self.eval(node.value, env, globals), self.eval(node.slice, env, globals)
            if isinstance(owner, Tensor):
                return self.index(owner, key, node)
            if type(owner) in (tuple, list, dict) and type(key) in (int, str, slice):
                return owner[key]
            method = inspect.getattr_static(type(owner), "__getitem__", None)
            if type(method) is FunctionType:
                self.record(node, "source_registry_lookup", key=render(key))
                return self.invoke(method, [owner, key], {}, node)
            raise SourceBoundary("Dynamic helper indexing is unsupported")
        if isinstance(node, ast.Call):
            function = self.eval(node.func, env, globals)
            args, kwargs = [], {}
            for item in node.args:
                value = self.eval(item.value if isinstance(item, ast.Starred) else item, env, globals)
                if isinstance(item, ast.Starred):
                    if type(value) not in (tuple, list) or len(value) > 128:
                        raise SourceBoundary("Helper star arguments are unresolved")
                    args.extend(value)
                else:
                    args.append(value)
            for keyword in node.keywords:
                value = self.eval(keyword.value, env, globals)
                additions = value if keyword.arg is None else {keyword.arg: value}
                if type(additions) is not dict or any(type(k) is not str for k in additions) or kwargs.keys() & additions.keys():
                    raise SourceBoundary("Helper keyword expansion is unresolved or duplicated")
                kwargs.update(additions)
            return self.invoke(function, args, kwargs, node)
        raise SourceBoundary(f"Unsupported helper expression: {type(node).__name__}")

    def binary(self, left: Any, right: Any, op: ast.AST) -> Any:
        if isinstance(op, ast.Add) and type(left) in (list, tuple) and type(left) is type(right) and len(left) + len(right) <= 128:
            return self.container(left + right)
        if isinstance(op, ast.Mult):
            sequence, count = (left, right) if type(left) in (list, tuple) else (right, left)
            if type(sequence) in (list, tuple) and type(count) is int and 0 <= count <= 128 and len(sequence) * count <= 128:
                return self.container(sequence * count)
        if isinstance(op, ast.BitXor):
            if type(left) is bool and type(right) is bool:
                return left ^ right
            raise SourceBoundary("Only proven Python booleans support static XOR")
        operations = {ast.Add: (operator.add, "+"), ast.Sub: (operator.sub, "-"), ast.Mult: (operator.mul, "*"),
                      ast.FloorDiv: (operator.floordiv, "//"), ast.BitAnd: (operator.and_, "&"), ast.BitOr: (operator.or_, "|")}
        if type(op) not in operations:
            raise SourceBoundary("Unsupported helper arithmetic")
        function, symbol = operations[type(op)]
        if isinstance(op, ast.FloorDiv) and type(right) in (int, float) and right == 0:
            raise SourceBoundary("Symbolic division by zero is invalid")
        if type(left) in (int, float, bool) and type(right) in (int, float, bool):
            result = function(left, right)
            if type(result) is int and result.bit_length() > 64:
                raise SourceBoundary("Static arithmetic exceeds the integer budget")
            return result
        if not all(isinstance(value, (Tensor, Symbol)) or type(value) in (int, float, bool) for value in (left, right)):
            raise SourceBoundary("Helper arithmetic operands are unsupported")
        expr = f"({expression(left)} {symbol} {expression(right)})"
        if len(expr) > 16384:
            raise SourceBoundary("Symbolic expression exceeds its budget")
        tensors = [value for value in (left, right) if isinstance(value, Tensor)]
        if tensors:
            if any(tensor.dtype != tensors[0].dtype for tensor in tensors):
                raise SourceBoundary("Mixed tensor arithmetic dtypes are not established")
            if isinstance(op, (ast.BitAnd, ast.BitOr)) and tensors[0].dtype != "BOOL":
                raise SourceBoundary("Only boolean tensor bitwise composition is supported")
            if isinstance(op, (ast.BitAnd, ast.BitOr)) and any(not isinstance(value, Tensor) and type(value) is not bool for value in (left, right)):
                raise SourceBoundary("Boolean bitwise scalar promotion is not established")
            if not isinstance(op, (ast.BitAnd, ast.BitOr)) and tensors[0].dtype == "BOOL":
                raise SourceBoundary("Boolean arithmetic promotion is unsupported")
            if tensors[0].dtype in {"I32", "I64"} and any(type(value) is float for value in (left, right)):
                raise SourceBoundary("Integer and floating scalar promotion is unsupported")
        return Tensor(self.broadcast(tensors), tensors[0].dtype, expr) if tensors else Symbol(expr)

    @staticmethod
    def broadcast(values: list[Tensor]) -> tuple[Any, ...]:
        result: tuple[Any, ...] = ()
        for value in values:
            width = max(len(result), len(value.shape))
            a, b = (1,) * (width - len(result)) + result, (1,) * (width - len(value.shape)) + value.shape
            if any(x != y and x != 1 and y != 1 for x, y in zip(a, b)):
                raise SourceBoundary("Symbolic tensor broadcasting is not established")
            result = tuple(y if x == 1 else x for x, y in zip(a, b))
        return result

    def index(self, tensor: Tensor, key: Any, node: ast.AST) -> Tensor:
        keys = key if type(key) is tuple else (key,)
        shape, consumed = [], 0
        for part in keys:
            if part is None:
                shape.append(1)
                continue
            if consumed >= len(tensor.shape):
                raise SourceBoundary("Tensor index exceeds its rank")
            dimension = tensor.shape[consumed]
            consumed += 1
            if type(part) is slice:
                if part.start in (None, 0) and part.stop is None and part.step in (None, 1):
                    shape.append(dimension)
                elif part.start in (None, 0) and type(part.stop) is int and part.stop >= 0 and part.step in (None, 1):
                    # Symbolic sequence lengths may be shorter than the slice bound.
                    shape.append(min(dimension, part.stop) if type(dimension) is int else Symbol(f"min({expression(dimension)}, {part.stop})"))
                else:
                    raise SourceBoundary("Symbolic slice length is not established")
            elif isinstance(part, Tensor):
                if part.shape:
                    raise SourceBoundary("Advanced vector indexing is unsupported")
            elif type(part) is not int:
                raise SourceBoundary("Tensor index type is not established")
        result = Tensor(tuple(shape) + tensor.shape[consumed:], tensor.dtype, f"index({tensor.expression})")
        self.record(node, "symbolic_tensor_index", output=result.spec())
        return result

    def method(self, owner: Any, name: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if isinstance(owner, Tensor):
            if name == "to":
                if any(key not in {"device", "dtype"} for key in kwargs) or len(args) > 1:
                    raise SourceBoundary("Unsupported tensor conversion arguments")
                if args and type(args[0]) is not torch.dtype and not (isinstance(args[0], Symbol) and args[0].expression == "input_device"):
                    raise SourceBoundary("Tensor conversion target is not established")
                dtype = kwargs.get("dtype", args[0] if args and type(args[0]) is torch.dtype else None)
                result = Tensor(owner.shape, dtype_name(dtype) if dtype is not None else owner.dtype, owner.expression)
            elif name in {"float", "half"} and not args and not kwargs:
                result = Tensor(owner.shape, "F32" if name == "float" else "F16", owner.expression)
            elif name in {"expand", "view", "reshape", "transpose", "permute", "unsqueeze", "squeeze", "flatten", "contiguous"}:
                from .ir import TensorSpec
                from .shape_rules import transform

                dims = list(args[0]) if len(args) == 1 and type(args[0]) in (list, tuple) else list(args)
                if kwargs:
                    raise SourceBoundary("Layout keyword arguments are not yet bound")
                check = transform(TensorSpec([render(d) for d in owner.shape], owner.dtype, "helper_input"),
                                  name, [render(d) for d in dims])
                if check.status not in {"proven", "conditional"}:
                    raise SourceBoundary(check.reason)
                for condition in check.constraints:
                    self.conditions.append({"condition": condition, "resolved": False, "value": None,
                                            "basis": "required_input_constraint", **self.current})
                self.record(node, "tensor_layout_contract", shape_check=check.record())
                result = Tensor(tuple(Symbol(d) if type(d) is str else d for d in check.spec.shape),
                                owner.dtype, f"{name}({owner.expression})")
            elif name == "cumsum" and len(args) == 1 and type(args[0]) is int and not kwargs:
                result = Tensor(owner.shape, "I64" if owner.dtype in {"BOOL", "I32", "I64"} else owner.dtype, f"cumsum({owner.expression}, {args[0]})")
            elif name == "new_ones" and args == [()] and set(kwargs) <= {"dtype"}:
                result = Tensor((), dtype_name(kwargs["dtype"]) if "dtype" in kwargs else owner.dtype, "1")
            elif name == "size" and not kwargs:
                return owner.shape if not args else owner.shape[args[0]]
            elif name == "dim" and not args and not kwargs:
                return len(owner.shape)
            else:
                raise SourceBoundary(f"Unsupported tensor method: {name}")
            self.record(node, f"tensor_{name}", output=result.spec())
            return result
        if type(owner) in (dict, list) and id(owner) in self.owned:
            if name == "pop" and type(owner) is dict and 1 <= len(args) <= 2 and type(args[0]) is str and not kwargs:
                return owner.pop(*args)
            if name == "extend" and type(owner) is list and len(args) == 1 and type(args[0]) in (tuple, list) and not kwargs and len(owner) + len(args[0]) <= 128:
                owner.extend(args[0])
                return None
        method = inspect.getattr_static(type(owner), name, None)
        if type(method) is FunctionType:
            return self.invoke(method, [owner, *args], kwargs, node)
        raise SourceBoundary(f"Unsupported helper method or mutation: {name}")

    def primitive(self, function: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if function is callable and len(args) == 1 and not kwargs:
            if isinstance(args[0], (Closure, Mapped, Bound)) or type(args[0]) in (FunctionType, BuiltinFunctionType, MethodType):
                return True
            if isinstance(args[0], Tensor) or type(args[0]) in (type(None), bool, int, float, str, tuple, list, dict):
                return False
            raise SourceBoundary("Callable status of a foreign object is unsupported")
        if function is all and len(args) == 1 and type(args[0]) in (tuple, list) and all(type(item) is bool for item in args[0]) and not kwargs:
            return all(args[0])
        if function is len and len(args) == 1 and type(args[0]) in (tuple, list, dict) and not kwargs:
            return len(args[0])
        if function is hasattr and len(args) == 2 and type(args[1]) is str and not kwargs:
            try:
                self.attribute(args[0], args[1])
                return True
            except MissingAttribute:
                return False
        if function is getattr and 2 <= len(args) <= 3 and type(args[1]) is str and not kwargs:
            try:
                return self.attribute(args[0], args[1])
            except MissingAttribute:
                if len(args) == 3:
                    return args[2]
                raise SourceBoundary("Required static attribute is missing")
        if function is isinstance and len(args) == 2 and not kwargs:
            value, targets = args
            targets = targets if type(targets) is tuple else (targets,)
            if any(target is not torch.Tensor and type(target) is not type for target in targets):
                raise SourceBoundary("Foreign isinstance checks are unsupported")
            if not isinstance(value, Tensor) and type(value) not in (type(None), int, bool, float, str, list, tuple, dict):
                raise SourceBoundary("Instance type is not statically established")
            mro = torch.Tensor.__mro__ if isinstance(value, Tensor) else type(value).__mro__
            return any(any(target is base for base in mro) for target in targets)
        if function is TORCH_PRIMITIVES["ones"] or function is TORCH_PRIMITIVES["zeros"]:
            dims = tuple(args[0]) if len(args) == 1 and type(args[0]) in (list, tuple) else tuple(args)
            if not dims or set(kwargs) - {"device", "dtype"}:
                raise SourceBoundary("Tensor factory arguments are unresolved")
            dtype = dtype_name(kwargs["dtype"]) if "dtype" in kwargs else dtype_name(torch.get_default_dtype())
            result = Tensor(dims, dtype, f"{function.__name__}({', '.join(expression(d) for d in dims)})")
            self.record(node, "tensor_factory", fill=1 if function is TORCH_PRIMITIVES["ones"] else 0,
                        dtype_basis="explicit dtype" if "dtype" in kwargs else "torch.get_default_dtype metadata", output=result.spec())
        elif function is TORCH_PRIMITIVES["arange"]:
            if len(args) != 1 or any(k not in {"device", "dtype"} for k in kwargs):
                raise SourceBoundary("Only zero-based one-bound arange is supported")
            if not isinstance(args[0], Symbol) and type(args[0]) is not int:
                raise SourceBoundary("Range extent is unresolved")
            result = Tensor((args[0],), dtype_name(kwargs["dtype"]) if "dtype" in kwargs else "I64", f"arange({expression(args[0])})")
        elif function is TORCH_PRIMITIVES["diff"]:
            if len(args) != 1 or not isinstance(args[0], Tensor) or set(kwargs) != {"prepend", "dim"}:
                raise SourceBoundary("Unsupported symbolic diff signature")
            value, prepend, dim = args[0], kwargs["prepend"], kwargs["dim"]
            if not isinstance(prepend, Tensor) or type(dim) is not int or not -len(value.shape) <= dim < len(value.shape):
                raise SourceBoundary("Diff axis or prepend is unresolved")
            axis = dim % len(value.shape)
            if len(value.shape) != len(prepend.shape) or any(a != b for i, (a, b) in enumerate(zip(value.shape, prepend.shape)) if i != axis):
                raise SourceBoundary("Diff prepend dimensions are incompatible")
            # For integer n >= 0: max(n + min(n, 1) - 1, 0) == n.
            extent = prepend.shape[axis]
            if extent != 1 and not (isinstance(extent, Symbol) and extent.expression == f"min({expression(value.shape[axis])}, 1)"):
                raise SourceBoundary("Diff output length is not established")
            result = Tensor(value.shape, value.dtype, f"diff({value.expression}, prepend={prepend.expression})")
        elif function is TORCH_PRIMITIVES["vmap"]:
            if len(args) != 1 or set(kwargs) != {"in_dims", "out_dims"} or kwargs["out_dims"] != 0 or type(kwargs["in_dims"]) is not tuple:
                raise SourceBoundary("Unsupported vmap dimensions")
            self.record(node, "symbolic_vmap", in_dims=list(kwargs["in_dims"]), out_dims=0)
            return Mapped(args[0], kwargs["in_dims"])
        elif function is TORCH_PRIMITIVES["tensor"]:
            if len(args) != 1 or type(args[0]) not in (int, float, bool) or set(kwargs) - {"device", "dtype"} or "dtype" not in kwargs:
                raise SourceBoundary("Only typed scalar tensor literals are supported")
            result = Tensor((), dtype_name(kwargs["dtype"]), repr(args[0]))
        elif function is TORCH_PRIMITIVES["finfo"] and len(args) == 1 and not kwargs:
            dtype = dtype_name(args[0])
            if dtype not in {"F16", "BF16", "F32", "F64"}:
                raise SourceBoundary("finfo requires a floating dtype")
            return FloatInfo(dtype)
        elif function is TORCH_PRIMITIVES["where"]:
            if len(args) != 3 or kwargs or not isinstance(args[0], Tensor) or args[0].dtype != "BOOL" or not isinstance(args[1], Tensor):
                raise SourceBoundary("Unsupported typed where signature")
            if not isinstance(args[2], Symbol) or args[2].expression != f"finfo({args[1].dtype}).min":
                raise SourceBoundary("where scalar dtype compatibility is not established")
            result = Tensor(self.broadcast([v for v in args if isinstance(v, Tensor)]), args[1].dtype,
                            f"where({args[0].expression}, {args[1].expression}, {args[2].expression})")
        else:
            context_module = sys.modules.get("torch._dynamo._trace_wrapped_higher_order_op")
            context = vars(context_module).get("TransformGetItemToIndex") if isinstance(context_module, ModuleType) else None
            if context is not None and function is context and not args and not kwargs:
                self.record(node, "symbolic_index_context", purpose="Interpret scalar tensor indices without materializing tensors")
                return ("symbolic_index_context",)
            return NotImplemented
        self.record(node, "torch_symbolic_primitive", callable=f"torch.{function.__name__}", output=result.spec())
        return result

    def vectorized(self, mapped: Mapped, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Tensor:
        if kwargs or len(args) != len(mapped.dimensions):
            raise SourceBoundary("vmap argument count does not match its axes")
        extent = None
        elements = []
        for value, dim in zip(args, mapped.dimensions):
            if dim is None:
                elements.append(value)
                continue
            if not isinstance(value, Tensor) or type(dim) is not int or not -len(value.shape) <= dim < len(value.shape):
                raise SourceBoundary("vmap input rank or dimension is invalid")
            dim %= len(value.shape)
            if extent is not None and extent != value.shape[dim]:
                raise SourceBoundary("vmap input extents disagree")
            extent = value.shape[dim]
            elements.append(Tensor(value.shape[:dim] + value.shape[dim + 1:], value.dtype, f"element({value.expression})"))
        if extent is None:
            raise SourceBoundary("vmap has no mapped axis")
        result = self.invoke(mapped.function, elements, {}, node)
        if not isinstance(result, Tensor):
            raise SourceBoundary("vmap helper output is not a tensor")
        return Tensor((extent, *result.shape), result.dtype, result.expression)
