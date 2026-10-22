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

from .static_tensors import TensorFactoryFact, analyze_tensor_factory_call

UNKNOWN = object()


@dataclass
class StaticValue:
    refs: list[tuple[str, str]] = field(default_factory=list)
    constant: Any = UNKNOWN
    module_path: str | None = None
    method: str | None = None
    items: list[StaticValue] | None = None
    fields: dict[str, StaticValue] | None = None
    # True when this value is a tensor produced by a factory. It is not a
    # dataflow reference and it is distinct from None.
    materialized: bool = False
    # Rank of a constructed tensor, when the factory size is a tuple or list.
    rank: int | None = None
    # Dependencies from opaque operations do not establish the result identity.
    identity_unknown: bool = False


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


def combine(values: list[StaticValue], preserve_tensor_facts: bool = False) -> StaticValue:
    result = StaticValue(list(dict.fromkeys(ref for value in values for ref in value.refs)))
    result.identity_unknown = any(value.identity_unknown for value in values)
    if preserve_tensor_facts:
        if any(value.constant is None for value in values) and not all(
            value.constant is None for value in values
        ):
            result.identity_unknown = True
        if values and all(value.materialized for value in values):
            result.materialized = True
        if values and all(value.rank is not None and value.rank == values[0].rank for value in values):
            result.rank = values[0].rank
    return result


def merge_outputs(values: list[StaticValue]) -> StaticValue:
    if len(values) == 1:
        return values[0]
    result = combine(values, preserve_tensor_facts=True)
    field_names = {name for value in values for name in (value.fields or {})}
    if field_names:
        result.fields = {
            name: combine(
                [value.fields[name] for value in values if value.fields and name in value.fields],
                preserve_tensor_facts=True,
            )
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

    def analyze(self) -> StaticFlow:
        try:
            method = inspect.unwrap(type(self.module).forward)
            self.source_globals = method.__globals__
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            function = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        except (OSError, TypeError, SyntaxError, StopIteration) as exc:
            self.flow.warnings.append(f"Forward source unavailable: {type(exc).__name__}")
            return self.flow
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
        for argument in function.args.args + function.args.kwonlyargs:
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
        self.statements(function.body)
        self.flow.outputs = merge_outputs(self.returned)
        return self.flow

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
                return True
            elif isinstance(node, ast.Expr):
                self.expression(node.value)
            elif isinstance(node, ast.If):
                condition = self.expression(node.test).constant
                if condition is not UNKNOWN:
                    if self.statements(node.body if condition else node.orelse):
                        return True
                else:
                    # Do not speculate through dynamic guards. Preserve dependencies of assignments,
                    # but explicitly mark module calls reached on either possible branch.
                    before, start = dict(self.env), len(self.flow.events)
                    left_returns = self.statements(node.body)
                    left = dict(self.env)
                    self.env = dict(before)
                    right_returns = self.statements(node.orelse)
                    assignments_changed = before != left or before != self.env
                    if left_returns != right_returns:
                        self.env = self.env if left_returns else left
                    else:
                        for name in left.keys() | self.env.keys():
                            a, b = left.get(name, StaticValue()), self.env.get(name, StaticValue())
                            self.env[name] = a if a == b else combine([a, b], preserve_tensor_facts=True)
                    for event in self.flow.events[start:]:
                        self.mark_conditional(event, ast.unparse(node.test))
                    if len(self.flow.events) > start or left_returns or right_returns or assignments_changed:
                        self.flow.warnings.append(
                            f"Conditional source path is unresolved: {ast.unparse(node.test)}"
                        )
                    if left_returns and right_returns:
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
                    self.flow.warnings.append(f"Loop could not be statically resolved at line {node.lineno}")
            elif isinstance(node, (ast.With, ast.Try)):
                if self.statements(node.body):
                    return True
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

    def constructed_tensor(self, fact: TensorFactoryFact, values: list[StaticValue]) -> StaticValue:
        result = combine(values)
        result.materialized = True
        result.rank = fact.rank
        result.identity_unknown = False
        assumption = fact.assumption
        if assumption not in self.flow.profile_assumptions:
            self.flow.profile_assumptions.append(assumption)
        return result

    def attribute(self, value: StaticValue, name: str) -> StaticValue:
        if name in {"shape", "dtype", "device", "ndim"}:
            return StaticValue()
        if value.module_path is not None:
            path = f"{value.module_path}.{name}" if value.module_path else name
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
        if isinstance(node, ast.Starred):
            return self.expression(node.value)
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
            return combine([self.expression(value) for value in node.values])
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
            value = self.expression(node.operand).constant
            if type(value) in (int, float):
                return StaticValue(constant=-value if isinstance(node.op, ast.USub) else value)
            return StaticValue()
        if isinstance(node, ast.BoolOp):
            values = [self.expression(item) for item in node.values]
            constants = [v.constant for v in values]
            if isinstance(node.op, ast.And) and any(c is not UNKNOWN and not c for c in constants):
                return StaticValue(constant=False)
            if isinstance(node.op, ast.Or) and any(c is not UNKNOWN and bool(c) for c in constants):
                return StaticValue(constant=True)
            if all(c is not UNKNOWN for c in constants):
                return StaticValue(
                    constant=all(constants) if isinstance(node.op, ast.And) else any(constants)
                )
            return StaticValue()
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.expression(node.left), self.expression(node.comparators[0])
            if isinstance(node.ops[0], (ast.Is, ast.IsNot)) and right.constant is None:
                if (
                    left.refs or left.module_path is not None or left.materialized
                ) and not left.identity_unknown:
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
            return (
                self.expression(node.body if condition else node.orelse)
                if condition is not UNKNOWN
                else combine(
                    [self.expression(node.body), self.expression(node.orelse)],
                    preserve_tensor_facts=True,
                )
            )
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
            if isinstance(node.func, ast.Attribute) and name in {"size", "dim", "numel"}:
                if name == "dim" and receiver is not None and receiver.rank is not None:
                    return StaticValue(constant=receiver.rank)
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
                known_tensor = bool(
                    (receiver.refs or receiver.materialized)
                    and receiver.items is None
                    and receiver.fields is None
                    and not receiver.identity_unknown
                )
                shape_changes = {
                    "view",
                    "reshape",
                    "transpose",
                    "permute",
                    "unsqueeze",
                    "squeeze",
                    "expand",
                    "flatten",
                }
                if name not in shape_changes:
                    return receiver
                return StaticValue(
                    refs=list(receiver.refs),
                    materialized=known_tensor,
                    rank=None,
                    identity_unknown=not known_tensor,
                )
            if name in {"enumerate", "iter"} and node.args:
                return self.expression(node.args[0])
            arguments = {str(i): self.expression(arg) for i, arg in enumerate(node.args)}
            arguments.update(
                {kw.arg: self.expression(kw.value) for kw in node.keywords if kw.arg is not None}
            )
            predicate = self.canonical_predicate(node, arguments)
            if predicate is not None:
                return predicate
            argument_values = {
                id(argument): arguments[str(index)] for index, argument in enumerate(node.args)
            }
            argument_values.update(
                {
                    id(keyword.value): arguments[keyword.arg]
                    for keyword in node.keywords
                    if keyword.arg is not None and keyword.arg in arguments
                }
            )

            def value_for(expression: ast.expr) -> StaticValue | None:
                if id(expression) in argument_values:
                    return argument_values[id(expression)]
                if isinstance(expression, ast.Name):
                    return self.env.get(expression.id)
                return None

            def known_size_rank(expression: ast.expr) -> int | None:
                value = value_for(expression)
                if value is None or value.items is None:
                    return None
                if any(type(item.constant) is int and item.constant < 0 for item in value.items):
                    return None
                return len(value.items)

            def known_tensor_fact(expression: ast.expr) -> tuple[bool, int | None]:
                value = value_for(expression)
                is_tensor = bool(
                    value
                    and value.items is None
                    and value.fields is None
                    and not value.identity_unknown
                    and (value.materialized or value.refs)
                )
                return is_tensor, value.rank if is_tensor else None

            factory = analyze_tensor_factory_call(
                node,
                {name: value for name, value in self.source_globals.items() if name not in self.env},
                bool(
                    receiver
                    and receiver.items is None
                    and receiver.fields is None
                    and (receiver.refs or receiver.materialized)
                    and not receiver.identity_unknown
                ),
                known_size_rank,
                known_tensor_fact,
            )
            if factory is not None:
                if factory.native:
                    factory_values = list(arguments.values()) + [
                        self.expression(keyword.value) for keyword in node.keywords if keyword.arg is None
                    ]
                    return self.constructed_tensor(factory, factory_values)
                self.flow.warnings.append(
                    f"Tensor factory unresolved: {ast.unparse(node.func)} ({factory.reason})"
                )
                dependencies = list(arguments.values()) + [
                    self.expression(keyword.value) for keyword in node.keywords if keyword.arg is None
                ]
                result = combine(dependencies)
                result.identity_unknown = True
                return result
            if function.module_path is not None and function.method is None:
                return self.event(node, function.module_path, "module", function.module_path, arguments)
            if function.method is not None and self.depth < 3:
                if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                    keyword.arg is None for keyword in node.keywords
                ):
                    self.flow.warnings.append(
                        f"Helper arguments could not be statically bound: {ast.unparse(node.func)}"
                    )
                    expanded = list(arguments.values()) + [
                        self.expression(keyword.value) for keyword in node.keywords if keyword.arg is None
                    ]
                    return combine(expanded)
                return self.helper(function.module_path, function.method, arguments, node)
            if name == "apply_chunking_to_forward" and node.args:
                if any(isinstance(arg, ast.Starred) for arg in node.args) or node.keywords:
                    self.flow.warnings.append(
                        f"Helper arguments could not be statically bound: {ast.unparse(node.func)}"
                    )
                    expanded = list(arguments.values()) + [
                        self.expression(keyword.value) for keyword in node.keywords if keyword.arg is None
                    ]
                    return combine(expanded)
                helper = arguments["0"]
                if helper.method:
                    return self.helper(
                        helper.module_path,
                        helper.method,
                        {str(i - 3): value for i, value in enumerate(arguments.values()) if i >= 3},
                        node,
                    )
            if name.startswith("create_") and name.endswith("mask"):
                return StaticValue()
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
                    result.identity_unknown = True
                    self.flow.warnings.append(
                        f"Opaque call {ast.unparse(node.func)}: dependencies retained, transformation unverified"
                    )
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
                result.identity_unknown = True
                self.flow.warnings.append(
                    f"Opaque call {ast.unparse(node.func)}: dependencies retained, transformation unverified"
                )
            return result
        return StaticValue()

    def helper(
        self,
        path: str | None,
        method_name: str,
        arguments: dict[str, StaticValue],
        node: ast.AST,
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
            return combine(list(arguments.values()))
        descriptor = inspect.getattr_static(type(module), method_name)
        if isinstance(descriptor, classmethod):
            self.flow.warnings.append(
                f"Helper arguments could not be statically bound: class method {path}.{method_name}"
            )
            return combine(list(arguments.values()))

        def unresolved() -> StaticValue:
            self.flow.warnings.append(
                f"Helper arguments could not be statically bound: {ast.unparse(node.func)}"
            )
            return combine(list(arguments.values()))

        is_static = isinstance(descriptor, staticmethod)
        positional_args = list(function.args.posonlyargs + function.args.args)
        receiver = None
        if not is_static and positional_args:
            receiver = positional_args.pop(0)
        keyword_only_args = list(function.args.kwonlyargs)
        positional_values = [
            arguments[str(index)] for index in range(len(arguments)) if str(index) in arguments
        ]
        named_values = {name: value for name, value in arguments.items() if not name.isdigit()}
        if len(positional_values) > len(positional_args) and function.args.vararg is None:
            return unresolved()

        provided: dict[str, StaticValue] = {}
        for parameter, value in zip(positional_args, positional_values):
            provided[parameter.arg] = value
        if function.args.vararg is not None:
            remaining = positional_values[len(positional_args) :]
            rest = combine(remaining)
            rest.items = remaining
            provided[function.args.vararg.arg] = rest

        positional_only = {parameter.arg for parameter in function.args.posonlyargs}
        if receiver is not None:
            positional_only.discard(receiver.arg)
        for name, value in named_values.items():
            parameter = next((item for item in positional_args + keyword_only_args if item.arg == name), None)
            if parameter is None or name in positional_only:
                if function.args.kwarg is None:
                    return unresolved()
                provided.setdefault(function.args.kwarg.arg, StaticValue(fields={})).fields[name] = value
            elif name in provided:
                return unresolved()
            else:
                provided[name] = value

        if function.args.kwarg is not None:
            fields = provided.get(function.args.kwarg.arg, StaticValue(fields={})).fields
            rest = combine(list(fields.values()))
            rest.fields = fields
            provided[function.args.kwarg.arg] = rest

        defaults = function.args.defaults
        positional_defaults = {
            parameter.arg for parameter in (positional_args[-len(defaults) :] if defaults else [])
        }
        keyword_defaults = {
            parameter.arg
            for parameter, default in zip(keyword_only_args, function.args.kw_defaults)
            if default is not None
        }
        for parameter in positional_args + keyword_only_args:
            if parameter.arg in provided:
                continue
            has_default = parameter.arg in positional_defaults or parameter.arg in keyword_defaults
            if not has_default:
                return unresolved()
            default = next(
                (
                    value
                    for candidate, value in zip(
                        positional_args[-len(defaults) :] if defaults else [], defaults
                    )
                    if candidate.arg == parameter.arg
                ),
                next(
                    (
                        value
                        for candidate, value in zip(keyword_only_args, function.args.kw_defaults)
                        if candidate.arg == parameter.arg
                    ),
                    None,
                ),
            )
            try:
                provided[parameter.arg] = StaticValue(constant=ast.literal_eval(default))
            except (ValueError, TypeError, SyntaxError):
                return unresolved()

        old_env, old_output, old_globals, old_returned = (
            self.env,
            self.flow.outputs,
            self.source_globals,
            self.returned,
        )
        self.source_globals = method.__globals__
        self.returned = []
        self.env = {receiver.arg: StaticValue(module_path=path)} if receiver is not None else {}
        self.env.update(provided)
        self.depth += 1
        self.statements(function.body)
        result = merge_outputs(self.returned)
        self.depth -= 1
        self.env, self.flow.outputs = old_env, old_output
        self.source_globals = old_globals
        self.returned = old_returned
        return result
