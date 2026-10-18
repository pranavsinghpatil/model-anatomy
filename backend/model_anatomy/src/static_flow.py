"""A bounded AST dependency interpreter that never evaluates model computations."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import math
import operator as arithmetic
import textwrap
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import torch
from transformers.utils.import_utils import is_torch_fx_proxy, is_torchdynamo_compiling

UNKNOWN = object()


@dataclass
class StaticValue:
    refs: list[tuple[str, str]] = field(default_factory=list)
    constant: Any = UNKNOWN
    module_path: str | None = None
    method: str | None = None
    items: list[StaticValue] | None = None
    fields: dict[str, StaticValue] | None = None


@dataclass
class FlowEvent:
    id: str
    module_path: str | None
    kind: str
    label: str
    arguments: dict[str, StaticValue]
    line: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticFlow:
    inputs: dict[str, StaticValue] = field(default_factory=dict)
    events: list[FlowEvent] = field(default_factory=list)
    outputs: StaticValue = field(default_factory=StaticValue)
    sequential_containers: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    profile_assumptions: list[dict[str, Any]] = field(default_factory=list)
    input_profile: list[dict[str, Any]] = field(default_factory=list)
    branch_conditions: list[dict[str, Any]] = field(default_factory=list)
    unexplained_operations: list[dict[str, Any]] = field(default_factory=list)
    call_sites: list[dict[str, Any]] = field(default_factory=list)
    source_call_sites: list[dict[str, Any]] = field(default_factory=list)


def combine(values: list[StaticValue]) -> StaticValue:
    return StaticValue(list(dict.fromkeys(ref for value in values for ref in value.refs)))


def merge_outputs(values: list[StaticValue]) -> StaticValue:
    if len(values) == 1:
        return values[0]
    result = combine(values)
    field_names = {name for value in values for name in (value.fields or {})}
    if field_names:
        result.fields = {
            name: combine([value.fields[name] for value in values if value.fields and name in value.fields])
            for name in field_names
        }
    return result


class StaticFlowAnalyzer:
    """Recover supported source dependencies and expose unsupported constructs honestly."""

    def __init__(self, model: torch.nn.Module, module_path: str, classifications: dict[str, Any]):
        self.model, self.path, self.classifications = model, module_path, classifications
        self.module = model.get_submodule(module_path) if module_path else model
        self.flow = StaticFlow()
        self.env: dict[str, StaticValue] = {"self": StaticValue(module_path=module_path)}
        self.depth = 0
        self.counter = 0
        self.source_globals: dict[str, Any] = {}
        self.returned: list[StaticValue] = []
        self.source_name = ""
        self.source_hash: str | None = None
        self.tainted_attributes: dict[str, StaticValue] = {}

    def analyze(self) -> StaticFlow:
        try:
            method = inspect.unwrap(type(self.module).forward)
            self.source_globals = method.__globals__
            source = textwrap.dedent(inspect.getsource(method))
            self.source_name = f"{method.__module__}.{method.__qualname__}"
            self.source_hash = hashlib.sha256(source.encode()).hexdigest()
            tree = ast.parse(source)
            function = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        except (OSError, TypeError, SyntaxError, StopIteration) as exc:
            self.flow.warnings.append(f"Forward source unavailable: {type(exc).__name__}")
            return self.flow
        self.collect_source_calls(function)
        optional = {
            "attention_mask",
            "encoder_attention_mask",
            "past_key_values",
            "past_key_value",
            "labels",
            "inputs_embeds",
            "decoder_inputs_embeds",
            "encoder_outputs",
            "bool_masked_pos",
            "token_type_ids",
        }
        disabled = {"use_cache", "output_attentions", "output_hidden_states", "interpolate_pos_encoding"}
        cross = (
            self.classifications[self.path].kind in {"decoder", "decoder_block", "cross_attention"}
            and self.model.config.is_encoder_decoder
        )
        for argument in function.args.posonlyargs + function.args.args + function.args.kwonlyargs:
            name = argument.arg
            if name == "self":
                continue
            if name in optional or name == "encoder_hidden_states" and not cross:
                self.env[name] = StaticValue(constant=None)
            elif name in disabled:
                self.env[name] = StaticValue(constant=False)
            elif name == "return_dict":
                self.env[name] = StaticValue(constant=True)
            elif (
                name in {"position_ids", "position_embeddings", "key_value_states", "encoder_hidden_states"}
                and (cross or self.classifications[self.path].kind in {"self_attention", "cross_attention"})
                or name
                in {
                    "input_ids",
                    "decoder_input_ids",
                    "pixel_values",
                    "input_features",
                    "input_values",
                    "hidden_states",
                    "hidden_state",
                    "input_tensor",
                    "input",
                    "x",
                    "position_bias",
                }
            ):
                self.env[name] = self.input(name)
            else:
                self.env[name] = StaticValue(constant=None)
            value = self.env[name]
            if value.constant is not UNKNOWN:
                self.flow.input_profile.append({
                    "argument": name, "value": value.constant,
                    "basis": "Canonical static profile; not a runtime observation",
                    **self.source_location(argument),
                })
        if function.args.vararg:
            self.env[function.args.vararg.arg] = StaticValue(items=[])
            self.flow.input_profile.append({
                "argument": function.args.vararg.arg, "value": [],
                "basis": "Canonical profile supplies no additional positional arguments",
                **self.source_location(function.args.vararg),
            })
        if function.args.kwarg:
            self.env[function.args.kwarg.arg] = StaticValue(fields={})
            self.flow.input_profile.append({
                "argument": function.args.kwarg.arg, "value": {},
                "basis": "Canonical profile supplies no additional keyword arguments",
                **self.source_location(function.args.kwarg),
            })
        self.statements(function.body)
        self.flow.outputs = merge_outputs(self.returned)
        return self.flow

    def source_location(self, node: ast.AST) -> dict[str, Any]:
        return {"line": getattr(node, "lineno", 0), "column": getattr(node, "col_offset", 0), "source": self.source_name,
                "source_sha256": self.source_hash}

    def collect_source_calls(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef, owner_path: str | None = None
    ) -> None:
        owner = self.path if owner_path is None else owner_path
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            # A helper method on a module is not an invocation of that module's forward.
            if isinstance(node.func, ast.Attribute) and node.func.attr != "forward":
                name = ast.unparse(node.func)
                if name.startswith("self.") and f"{owner}.{name[5:]}".lstrip(".") not in self.classifications:
                    continue
            candidates = []
            for part in ast.walk(node.func):
                if not isinstance(part, ast.Attribute):
                    continue
                name = ast.unparse(part)
                if name.startswith("self."):
                    path = f"{owner}.{name[5:]}".lstrip(".")
                    if path in self.classifications:
                        candidates.append(path)
            if candidates:
                self.flow.source_call_sites.append({"module_path": max(candidates, key=len),
                                                    "excluded_by_profile": False,
                                                    **self.source_location(node)})

    def exclude_source_calls(self, nodes: list[ast.AST], evidence: dict[str, Any]) -> None:
        locations = {(getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                     for statement in nodes for node in ast.walk(statement) if isinstance(node, ast.Call)}
        for call in self.flow.source_call_sites:
            if call["source"] == self.source_name and (call["line"], call["column"]) in locations:
                call["excluded_by_profile"] = True
                call["profile_condition"] = evidence

    def unsupported(self, node: ast.AST, reason: str,
                    values: list[StaticValue] | None = None) -> StaticValue:
        """Expose an opaque dependency boundary without interpreting its computation."""
        detail = {"construct": type(node).__name__, "expression": ast.unparse(node),
                  "reason": reason, **self.source_location(node)}
        self.flow.unexplained_operations.append(detail)
        self.flow.warnings.append(f"Unsupported {detail['construct']} at line {detail['line']}: {reason}")
        dependencies = combine(values if values is not None else list(self.env.values()))
        if isinstance(node, ast.stmt):
            detail["potential_module_paths"] = sorted({
                f"{self.path}.{ast.unparse(item)[5:]}".lstrip(".")
                for item in ast.walk(node) if isinstance(item, ast.Attribute)
                and ast.unparse(item).startswith("self.")
                and f"{self.path}.{ast.unparse(item)[5:]}".lstrip(".") in self.classifications
            })
        value = self.event(node, None, "unknown", "Unsupported source boundary",
                           {"dependencies": dependencies}, {"unsupported": detail})
        if isinstance(node, ast.stmt):
            for item in ast.walk(node):
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                    self.env[item.id] = value
                elif isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store):
                    name = ast.unparse(item)
                    if name.startswith("self."):
                        owner = self.env.get("self", StaticValue()).module_path or ""
                        attribute = name[5:].split(".", 1)[0]
                        self.tainted_attributes[f"{owner}.{attribute}".lstrip(".")] = value
            if any(isinstance(item, ast.Return) for item in ast.walk(node)):
                self.returned.append(value)
        return value

    def branch(self, node: ast.AST, value: Any) -> dict[str, Any]:
        evidence = {"condition": ast.unparse(node),
                    "resolved": value is not UNKNOWN,
                    "value": bool(value) if value is not UNKNOWN else None,
                    "basis": "Static attributes and canonical input profile",
                    **self.source_location(node)}
        self.flow.branch_conditions.append(evidence)
        return evidence

    def input(self, name: str) -> StaticValue:
        value = StaticValue([(f"input:{name}", "output")])
        self.flow.inputs[name] = value
        return value

    def event(
        self,
        node: ast.AST,
        path: str | None,
        kind: str,
        label: str,
        arguments: dict[str, StaticValue],
        metadata: dict[str, Any] | None = None,
    ) -> StaticValue:
        self.counter += 1
        identifier = f"event{self.counter}"
        self.flow.events.append(
            FlowEvent(identifier, path, kind, label, arguments, getattr(node, "lineno", 0), metadata or {})
        )
        return StaticValue([(identifier, "output")])

    def assign(self, target: ast.AST, value: StaticValue) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                self.assign(
                    item,
                    value.items[index]
                    if value.items and index < len(value.items)
                    else StaticValue([(ref, str(index)) for ref, _ in value.refs]),
                )
        else:
            boundary = self.unsupported(target, "Assignment target effects are unsupported", [value])
            root = target
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and root.id != "self":
                self.env[root.id] = boundary
            elif isinstance(root, ast.Name):
                name = ast.unparse(target)
                attribute = name[5:].split(".", 1)[0].split("[", 1)[0]
                owner = self.env.get("self", StaticValue()).module_path or ""
                self.tainted_attributes[f"{owner}.{attribute}".lstrip(".")] = boundary

    def statements(self, statements: list[ast.stmt]) -> bool:
        for index, node in enumerate(statements):
            if isinstance(node, ast.Assign):
                value = self.expression(node.value)
                for target in node.targets:
                    self.assign(target, value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self.assign(node.target, self.expression(node.value))
            elif isinstance(node, ast.AugAssign):
                self.assign(
                    node.target,
                    self.operation(node, self.expression(node.target), self.expression(node.value), node.op),
                )
            elif isinstance(node, ast.Return):
                self.flow.outputs = self.expression(node.value)
                self.returned.append(self.flow.outputs)
                self.exclude_source_calls(statements[index + 1:], {
                    "condition": "Unreachable after return", "resolved": True,
                    "value": False, **self.source_location(node),
                })
                return True
            elif isinstance(node, ast.Expr):
                self.expression(node.value)
            elif isinstance(node, ast.If):
                condition = self.expression(node.test).constant
                evidence = self.branch(node.test, condition)
                if condition is not UNKNOWN:
                    self.exclude_source_calls(node.orelse if condition else node.body, evidence)
                    start = len(self.flow.events)
                    returned = self.statements(node.body if condition else node.orelse)
                    for event in self.flow.events[start:]:
                        event.metadata.setdefault("branch_evidence", []).append(evidence)
                    if returned:
                        self.exclude_source_calls(statements[index + 1:], evidence)
                        return True
                else:
                    # Do not speculate through dynamic guards. Preserve dependencies of assignments,
                    # but explicitly mark module calls reached on either possible branch.
                    before, start = dict(self.env), len(self.flow.events)
                    left_returns = self.statements(node.body)
                    left = dict(self.env)
                    for event in self.flow.events[start:]:
                        self.mark_conditional(event, ast.unparse(node.test))
                        event.metadata.setdefault("branch_evidence", []).append(dict(evidence, arm="body"))
                    right_start = len(self.flow.events)
                    self.env = dict(before)
                    right_returns = self.statements(node.orelse)
                    assignments_changed = before != left or before != self.env
                    if left_returns != right_returns:
                        self.env = self.env if left_returns else left
                    else:
                        for name in left.keys() | self.env.keys():
                            a, b = left.get(name, StaticValue()), self.env.get(name, StaticValue())
                            self.env[name] = a if a == b else combine([a, b])
                    for event in self.flow.events[right_start:]:
                        self.mark_conditional(event, f"not ({ast.unparse(node.test)})")
                        event.metadata.setdefault("branch_evidence", []).append(dict(evidence, arm="else"))
                    if len(self.flow.events) > start or left_returns or right_returns or assignments_changed:
                        self.flow.warnings.append(
                            f"Conditional source path is unresolved: {ast.unparse(node.test)}"
                        )
                    if left_returns and right_returns:
                        self.exclude_source_calls(
                            statements[index + 1:], dict(evidence, basis="Both branches return")
                        )
                        return True
                    if left_returns or right_returns:
                        start = len(self.flow.events)
                        returned = self.statements(statements[index + 1 :])
                        condition = ast.unparse(node.test)
                        continuation = f"not ({condition})" if left_returns else condition
                        for event in self.flow.events[start:]:
                            self.mark_conditional(event, continuation)
                        return returned
            elif isinstance(node, ast.For):
                iterable = self.expression(node.iter)
                if iterable.module_path is not None:
                    target = node.target.elts[-1] if isinstance(node.target, ast.Tuple) else node.target
                    before = dict(self.env)
                    self.assign(target, iterable)
                    start = len(self.flow.events)
                    self.statements(node.body)
                    calls = [
                        event
                        for event in self.flow.events[start:]
                        if event.module_path == iterable.module_path
                    ]
                    for event in calls:
                        if any(
                            ref == event.id
                            for name, value in self.env.items()
                            if name in before
                            and any(
                                ref in before[name].refs
                                for arg in event.arguments.values()
                                for ref in arg.refs
                            )
                            for ref, _ in value.refs
                        ):
                            self.flow.sequential_containers.add(iterable.module_path)
                else:
                    self.unsupported(node, "Loop iterable and iteration effects could not be statically resolved")
                if node.orelse:
                    self.unsupported(node, "For/else completion and break conditions are not established")
            elif isinstance(node, ast.Pass):
                continue
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                self.unsupported(node, "Context manager entry/exit and exception effects are not established")
            elif isinstance(node, (ast.Try, ast.TryStar)):
                self.unsupported(node, "Try body, handlers, else and finally remain possible; exception flow is unsupported")
            else:
                self.unsupported(node, "Statement semantics are not supported by the static interpreter")
        return False

    @staticmethod
    def mark_conditional(event: FlowEvent, condition: str) -> None:
        existing = event.metadata.get("conditional")
        event.metadata["conditional"] = f"({existing}) and ({condition})" if existing else condition

    def operation(
        self, node: ast.AST, left: StaticValue, right: StaticValue, operator: ast.operator
    ) -> StaticValue:
        if not left.refs and not right.refs:
            operations = {
                ast.Add: arithmetic.add,
                ast.Sub: arithmetic.sub,
                ast.Mult: arithmetic.mul,
                ast.Div: arithmetic.truediv,
                ast.Pow: arithmetic.pow,
            }
            if (
                type(left.constant) in (int, float)
                and type(right.constant) in (int, float)
                and type(operator) in operations
                and abs(left.constant) <= 1e6
                and abs(right.constant) <= 8
            ):
                try:
                    value = operations[type(operator)](left.constant, right.constant)
                    if type(value) in (int, float) and math.isfinite(value):
                        return StaticValue(constant=value)
                except (OverflowError, ValueError, ZeroDivisionError):
                    pass
            return StaticValue()
        if not left.refs or not right.refs:
            if (
                left.refs
                and type(right.constant) in (int, float)
                and isinstance(operator, (ast.Mult, ast.Div))
            ):
                factor = (
                    right.constant
                    if isinstance(operator, ast.Mult)
                    else 1 / right.constant
                    if right.constant
                    else None
                )
                return self.event(
                    node, None, "scale", "Scale hidden states", {"input": left}, {"factor": factor}
                )
            return combine([left, right])
        kind = (
            "residual"
            if isinstance(operator, ast.Add)
            else "gating"
            if isinstance(operator, ast.Mult)
            else "operation"
        )
        label = (
            "Residual Add"
            if kind == "residual"
            else "Elementwise gate"
            if kind == "gating"
            else type(operator).__name__
        )
        metadata = {}
        if kind == "residual":
            left_ast = node.left if isinstance(node, ast.BinOp) else node.target
            right_ast = node.right if isinstance(node, ast.BinOp) else node.value
            for port, expression in (("left", left_ast), ("right", right_ast)):
                if isinstance(expression, ast.Name) and expression.id in {
                    "residual",
                    "input_tensor",
                    "shortcut",
                }:
                    metadata["skip_argument"] = port
        return self.event(node, None, kind, label, {"left": left, "right": right}, metadata)

    def attribute(self, value: StaticValue, name: str) -> StaticValue:
        if name in {"shape", "dtype", "device", "ndim"}:
            return StaticValue()
        if value.module_path is not None:
            path = f"{value.module_path}.{name}" if value.module_path else name
            if path in self.tainted_attributes:
                return self.tainted_attributes[path]
            if path in self.classifications:
                return StaticValue(module_path=path)
            module = self.model.get_submodule(value.module_path) if value.module_path else self.model
            attribute = getattr(module, name, UNKNOWN)
            if isinstance(attribute, torch.Tensor):
                return StaticValue([(f"parameter:{path}", "output")])
            if inspect.ismethod(attribute) or inspect.isfunction(attribute):
                return StaticValue(module_path=value.module_path, method=name)
            return StaticValue(constant=attribute)
        if value.constant is not UNKNOWN and value.constant is not None:
            return StaticValue(constant=getattr(value.constant, name, UNKNOWN))
        return StaticValue([(ref, name) for ref, _ in value.refs])

    def canonical_predicate(self, node: ast.Call, arguments: dict[str, StaticValue]) -> StaticValue | None:
        """Resolve native predicates by identity, never by the spelling of a call."""
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return None
        parts = []
        expression = node.func
        while isinstance(expression, ast.Attribute):
            parts.append(expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name) or expression.id in self.env:
            return None
        function = self.source_globals.get(expression.id, vars(builtins).get(expression.id, UNKNOWN))
        for name in reversed(parts):
            if not isinstance(function, ModuleType):
                return None
            function = vars(function).get(name, UNKNOWN)
        if function is isinstance:
            if len(node.args) != 2 or node.keywords or not isinstance(node.args[1], ast.Name):
                return None
            target_name = node.args[1].id
            if target_name in self.env:
                return None
            target = self.source_globals.get(target_name, vars(builtins).get(target_name, UNKNOWN))
            value = arguments["0"].constant
            if any(target is container for container in (dict, list, tuple)) and type(value) in (
                type(None),
                bool,
                int,
                float,
                str,
            ):
                return StaticValue(constant=False)
            return None
        if function is is_torch_fx_proxy:
            value = arguments.get("0", arguments.get("x", StaticValue()))
            if len(arguments) != 1 or not (
                value.refs or type(value.constant) in (type(None), bool, int, float, str)
            ):
                return None
            basis = "Canonical inputs and source dependencies are tensors, not FX Proxy objects"
        elif any(
            function is predicate
            for predicate in (torch.jit.is_tracing, is_torchdynamo_compiling, torch.compiler.is_compiling)
        ):
            if arguments or node.args or node.keywords:
                return None
            basis = "Canonical static profile does not execute forward, JIT tracing or compilation"
        else:
            return None
        try:
            source_hash = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        except (OSError, TypeError):
            return None
        assumption = {
            "predicate": f"{function.__module__}.{function.__qualname__}",
            "source_sha256": source_hash,
            "value": False,
            "basis": basis,
        }
        if assumption not in self.flow.profile_assumptions:
            self.flow.profile_assumptions.append(assumption)
        return StaticValue(constant=False)

    def expression(self, node: ast.AST | None) -> StaticValue:
        if node is None:
            return StaticValue(constant=None)
        if isinstance(node, ast.Constant):
            return StaticValue(constant=node.value)
        if isinstance(node, ast.Name):
            return self.env.get(node.id, StaticValue())
        if isinstance(node, ast.NamedExpr):
            value = self.expression(node.value)
            self.assign(node.target, value)
            return value
        if isinstance(node, ast.Attribute):
            return self.attribute(self.expression(node.value), node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.expression(item) for item in node.elts]
            result = combine(values)
            result.items = values
            return result
        if isinstance(node, ast.Dict):
            fields = {}
            for key, value in zip(node.keys, node.values):
                if key is None or not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    return self.unsupported(node, "Dictionary keys or unpacking are not statically known")
                fields[key.value] = self.expression(value)
            result = combine(list(fields.values()))
            result.fields = fields
            return result
        if isinstance(node, ast.Slice):
            return StaticValue()
        if isinstance(node, ast.Subscript):
            if getattr(self.model.config, "model_type", None) == "vit" and isinstance(node.slice, ast.Tuple):
                value = self.expression(node.value)
                indices = []
                for item in node.slice.elts:
                    if isinstance(item, ast.Slice) and item.lower is item.upper is item.step is None:
                        indices.append(None)
                    elif isinstance(item, ast.Constant) and type(item.value) is int:
                        indices.append(item.value)
                    else:
                        indices.append("unknown")
                return self.event(
                    node, None, "selection", "Select tokens", {"input": value}, {"indices": indices}
                )
            value, index = self.expression(node.value), self.expression(node.slice).constant
            if value.module_path is not None and type(index) is int:
                module = self.model.get_submodule(value.module_path) if value.module_path else self.model
                names = list(module._modules)
                if -len(names) <= index < len(names):
                    return StaticValue(module_path=value.module_path + "." + names[index])
            if value.module_path is not None:
                return value
            if value.items and type(index) is int and -len(value.items) <= index < len(value.items):
                return value.items[index]
            return StaticValue(
                [(ref, str(index) if index is not UNKNOWN else port) for ref, port in value.refs]
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.expression(node.operand)
            return (
                StaticValue(constant=not value.constant) if value.constant is not UNKNOWN else StaticValue()
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand)
            if type(value.constant) in (int, float):
                return StaticValue(constant=-value.constant if isinstance(node.op, ast.USub) else value.constant)
            return self.unsupported(node, "Unary operation is not statically established", [value])
        if isinstance(node, ast.BoolOp):
            value = self.expression(node.values[0])
            for index, item in enumerate(node.values[1:], start=1):
                if type(value.constant) not in (bool, int, float, str, type(None)):
                    return self.unsupported(
                        node, "Short-circuit condition is unresolved; later operands remain possible",
                        [value, combine(list(self.env.values()))],
                    )
                if isinstance(node.op, ast.And) and not value.constant:
                    self.exclude_source_calls(node.values[index:], self.branch(node, value.constant))
                    return value
                if isinstance(node.op, ast.Or) and value.constant:
                    self.exclude_source_calls(node.values[index:], self.branch(node, value.constant))
                    return value
                value = self.expression(item)
            return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.expression(node.left), self.expression(node.comparators[0])
            if isinstance(node.ops[0], (ast.Is, ast.IsNot)) and right.constant is None:
                if left.refs or left.module_path is not None:
                    return StaticValue(constant=isinstance(node.ops[0], ast.IsNot))
                if left.constant is not UNKNOWN:
                    return StaticValue(constant=(left.constant is None) == isinstance(node.ops[0], ast.Is))
            if (
                left.constant is not UNKNOWN
                and right.constant is not UNKNOWN
                and type(left.constant) in (bool, int, str, type(None))
            ):
                if isinstance(node.ops[0], ast.Eq):
                    return StaticValue(constant=left.constant == right.constant)
                if isinstance(node.ops[0], ast.NotEq):
                    return StaticValue(constant=left.constant != right.constant)
            return StaticValue()
        if isinstance(node, ast.IfExp):
            condition = self.expression(node.test).constant
            evidence = self.branch(node.test, condition)
            if condition is not UNKNOWN:
                self.exclude_source_calls([node.orelse if condition else node.body], evidence)
                start = len(self.flow.events)
                value = self.expression(node.body if condition else node.orelse)
                for event in self.flow.events[start:]:
                    event.metadata.setdefault("branch_evidence", []).append(evidence)
                return value
            before = dict(self.env)
            start = len(self.flow.events)
            left = self.expression(node.body)
            left_env = dict(self.env)
            for event in self.flow.events[start:]:
                self.mark_conditional(event, ast.unparse(node.test))
                event.metadata.setdefault("branch_evidence", []).append(dict(evidence, arm="body"))
            self.env = dict(before)
            start = len(self.flow.events)
            right = self.expression(node.orelse)
            for event in self.flow.events[start:]:
                self.mark_conditional(event, f"not ({ast.unparse(node.test)})")
                event.metadata.setdefault("branch_evidence", []).append(dict(evidence, arm="else"))
            for name in left_env.keys() | self.env.keys():
                a, b = left_env.get(name, StaticValue()), self.env.get(name, StaticValue())
                self.env[name] = a if a == b else combine([a, b])
            self.flow.warnings.append(f"Conditional source path is unresolved: {ast.unparse(node.test)}")
            return merge_outputs([left, right])
        if isinstance(node, ast.BinOp):
            return self.operation(node, self.expression(node.left), self.expression(node.right), node.op)
        if isinstance(node, ast.Call):
            receiver = self.expression(node.func.value) if isinstance(node.func, ast.Attribute) else None
            function = (
                self.attribute(receiver, node.func.attr)
                if receiver is not None
                else self.expression(node.func)
            )
            name = ast.unparse(node.func).rsplit(".", 1)[-1]
            transparent_methods = {
                "size", "dim", "numel", "view", "reshape", "contiguous", "to",
                "type_as", "float", "half", "transpose", "permute", "unsqueeze",
                "squeeze", "expand", "flatten",
            }
            if (
                isinstance(node.func, ast.Attribute)
                and name in transparent_methods
                and (any(isinstance(arg, ast.Starred) for arg in node.args)
                     or any(keyword.arg is None for keyword in node.keywords))
            ):
                return self.unsupported(
                    node, "Star arguments for this tensor method cannot be safely bound",
                    [receiver or StaticValue()],
                )
            if isinstance(node.func, ast.Attribute) and name in {"size", "dim", "numel"}:
                return StaticValue()
            if isinstance(node.func, ast.Attribute) and name in {
                "view",
                "reshape",
                "contiguous",
                "to",
                "type_as",
                "float",
                "half",
                "transpose",
                "permute",
                "unsqueeze",
                "squeeze",
                "expand",
                "flatten",
            }:
                if getattr(self.model.config, "model_type", None) == "vit" and name in {
                    "view",
                    "reshape",
                    "transpose",
                    "permute",
                    "unsqueeze",
                    "squeeze",
                    "expand",
                    "flatten",
                }:
                    values = [self.expression(arg).constant for arg in node.args]
                    values = [
                        value
                        if type(value) is int
                        else "B"
                        if isinstance(arg, ast.Name)
                        and arg.id == "batch_size"
                        and type(self.module).__module__ == "transformers.models.vit.modeling_vit"
                        else "unknown"
                        for arg, value in zip(node.args, values)
                    ]
                    return self.event(
                        node,
                        None,
                        "reshape",
                        name.title(),
                        {"input": receiver},
                        {"method": name, "dimensions": values, "keyword_arguments": bool(node.keywords)},
                    )
                return receiver
            if name in {"enumerate", "iter"} and node.args:
                return self.expression(node.args[0])
            positional = []
            keywords = {}
            unresolved_unpacking = False
            for arg in node.args:
                value = self.expression(arg.value if isinstance(arg, ast.Starred) else arg)
                if isinstance(arg, ast.Starred):
                    if value.items is None or len(value.items) > 128:
                        unresolved_unpacking = True
                        positional.append(value)
                    else:
                        positional.extend(value.items)
                else:
                    positional.append(value)
            for keyword in node.keywords:
                value = self.expression(keyword.value)
                if keyword.arg is None:
                    if value.fields is None or len(value.fields) > 128:
                        unresolved_unpacking = True
                        positional.append(value)
                    elif keywords.keys() & value.fields.keys():
                        unresolved_unpacking = True
                    else:
                        keywords.update(value.fields)
                elif keyword.arg in keywords:
                    unresolved_unpacking = True
                else:
                    keywords[keyword.arg] = value
            arguments = {str(i): value for i, value in enumerate(positional)}
            arguments.update(keywords)
            if unresolved_unpacking:
                return self.unsupported(node, "Star arguments or keyword expansion cannot be safely bound",
                                        [receiver or StaticValue(), *arguments.values()])
            predicate = self.canonical_predicate(node, arguments)
            if predicate is not None:
                return predicate
            if function.module_path is not None and function.method is None:
                self.flow.call_sites.append({"module_path": function.module_path,
                                             **self.source_location(node)})
                return self.event(node, function.module_path, "module", function.module_path, arguments)
            if function.method is not None and self.depth < 3:
                return self.helper(function.module_path, function.method, positional, node, keywords)
            if name == "apply_chunking_to_forward" and node.args:
                helper = arguments["0"]
                if helper.method:
                    return self.helper(helper.module_path, helper.method, positional[3:], node, keywords)
            if name == "arange":
                return self.input("position_ids")
            if name in {"matmul", "bmm", "einsum", "cat", "stack"}:
                kind = "matmul" if name in {"matmul", "bmm", "einsum"} else "concatenation"
                dimension = arguments.get("dim", arguments.get("1", StaticValue(constant=0))).constant
                metadata = {"method": name}
                if name in {"cat", "stack"} and type(dimension) is int:
                    metadata["dimension"] = dimension
                return self.event(
                    node,
                    None,
                    kind,
                    "Matrix multiply" if kind == "matmul" else "Concatenate",
                    arguments,
                    metadata,
                )
            if name in {"hasattr", "isinstance"}:
                if name == "hasattr" and len(node.args) == 2:
                    obj, attribute = arguments["0"], arguments["1"].constant
                    if obj.module_path is not None and isinstance(attribute, str):
                        module = self.model.get_submodule(obj.module_path) if obj.module_path else self.model
                        return StaticValue(constant=hasattr(module, attribute))
                return StaticValue()
            if isinstance(node.func, ast.Attribute):
                result = combine([receiver, *arguments.values()])
                if result.refs:
                    self.flow.warnings.append(
                        f"Opaque call {ast.unparse(node.func)}: dependencies retained, transformation unverified"
                    )
                    return self.unsupported(node, "Opaque call transformation is unverified", [receiver, *arguments.values()])
                self.flow.unexplained_operations.append({"construct": "Call", "expression": ast.unparse(node),
                                                        "reason": "Opaque call without tensor dependencies", **self.source_location(node)})
                return result
            if name[0].isupper() and any(
                key in arguments
                for key in (
                    "logits",
                    "last_hidden_state",
                    "pooler_output",
                    "logits_per_image",
                    "logits_per_text",
                )
            ):
                fields = {
                    key: value
                    for key, value in arguments.items()
                    if key
                    in {
                        "logits",
                        "last_hidden_state",
                        "pooler_output",
                        "logits_per_image",
                        "logits_per_text",
                        "text_embeds",
                        "image_embeds",
                    }
                }
                result = combine(list(fields.values()))
                result.fields = fields
                return result
            result = combine(list(arguments.values()))
            if result.refs:
                self.flow.warnings.append(
                    f"Opaque call {ast.unparse(node.func)}: dependencies retained, transformation unverified"
                )
                return self.unsupported(node, "Opaque call transformation is unverified", list(arguments.values()))
            self.flow.unexplained_operations.append({"construct": "Call", "expression": ast.unparse(node),
                                                    "reason": "Opaque call without tensor dependencies", **self.source_location(node)})
            return result
        return self.unsupported(node, "Expression semantics are not supported by the static interpreter")

    def helper(
        self, path: str | None, method_name: str, values: list[StaticValue], node: ast.AST,
        keywords: dict[str, StaticValue] | None = None,
    ) -> StaticValue:
        module = self.model.get_submodule(path) if path else self.model
        try:
            method = inspect.unwrap(getattr(type(module), method_name))
            function = next(
                n
                for n in ast.parse(textwrap.dedent(inspect.getsource(method))).body
                if isinstance(n, ast.FunctionDef)
            )
        except (AttributeError, OSError, TypeError, SyntaxError, StopIteration):
            self.flow.warnings.append(f"Helper source unavailable: {path}.{method_name}; dependencies only")
            return self.unsupported(node, f"Helper source unavailable: {path}.{method_name}",
                                    [*values, *(keywords or {}).values()])
        bound = self.bind_helper(function, values, keywords or {})
        if bound is None:
            return self.unsupported(node, f"Helper arguments cannot be bound: {method_name}",
                                    [*values, *(keywords or {}).values()])
        old_source = self.source_name, self.source_hash
        self.source_name = f"{method.__module__}.{method.__qualname__}"
        self.source_hash = hashlib.sha256(textwrap.dedent(inspect.getsource(method)).encode()).hexdigest()
        self.collect_source_calls(function, path or "")
        old_env, old_output, old_globals, old_returned = (
            self.env,
            self.flow.outputs,
            self.source_globals,
            self.returned,
        )
        self.source_globals = method.__globals__
        self.returned = []
        self.env = {"self": StaticValue(module_path=path)}
        self.env.update(bound)
        self.depth += 1
        self.statements(function.body)
        result = merge_outputs(self.returned)
        self.depth -= 1
        self.env, self.flow.outputs = old_env, old_output
        self.source_globals = old_globals
        self.returned = old_returned
        self.source_name, self.source_hash = old_source
        return result

    @staticmethod
    def bind_helper(function: ast.FunctionDef, values: list[StaticValue],
                    keywords: dict[str, StaticValue]) -> dict[str, StaticValue] | None:
        """Bind AST parameter names, defaults and variadics without executing a helper."""
        args = function.args
        positional = args.posonlyargs + args.args
        if positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        if len(values) > len(positional) and args.vararg is None:
            return None
        bound = {arg.arg: value for arg, value in zip(positional, values)}
        allowed = {arg.arg for arg in args.args + args.kwonlyargs} - {"self", "cls"}
        extras = {}
        for name, value in keywords.items():
            if name in allowed:
                if name in bound:
                    return None
                bound[name] = value
            elif args.kwarg:
                extras[name] = value
            else:
                return None
        defaults = dict(zip([arg.arg for arg in positional][-len(args.defaults):], args.defaults)) if args.defaults else {}
        defaults.update({arg.arg: default for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is not None})
        for arg in positional + args.kwonlyargs:
            if arg.arg in bound:
                continue
            if arg.arg not in defaults:
                return None
            try:
                bound[arg.arg] = StaticValue(constant=ast.literal_eval(defaults[arg.arg]))
            except (ValueError, TypeError, SyntaxError):
                # Defaults referencing globals are unknown, never copied from the caller's environment.
                bound[arg.arg] = StaticValue()
        if args.vararg:
            bound[args.vararg.arg] = StaticValue(items=values[len(positional):])
        if args.kwarg:
            bound[args.kwarg.arg] = StaticValue(fields=extras)
        return bound
