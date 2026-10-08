"""A bounded AST dependency interpreter that never evaluates model computations."""

from __future__ import annotations

import ast
import inspect
import math
import operator as arithmetic
import textwrap
from dataclasses import dataclass, field
from typing import Any

import torch

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


def combine(values: list[StaticValue]) -> StaticValue:
    return StaticValue(list(dict.fromkeys(ref for value in values for ref in value.refs)))


class StaticFlowAnalyzer:
    """Recover supported source dependencies and expose unsupported constructs honestly."""

    def __init__(self, model: torch.nn.Module, module_path: str, classifications: dict[str, Any]):
        self.model, self.path, self.classifications = model, module_path, classifications
        self.module = model.get_submodule(module_path) if module_path else model
        self.flow = StaticFlow()
        self.env: dict[str, StaticValue] = {"self": StaticValue(module_path=module_path)}
        self.depth = 0
        self.counter = 0

    def analyze(self) -> StaticFlow:
        try:
            method = type(self.module).forward
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

    def statements(self, statements: list[ast.stmt]) -> None:
        for node in statements:
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
                return
            elif isinstance(node, ast.Expr):
                self.expression(node.value)
            elif isinstance(node, ast.If):
                condition = self.expression(node.test).constant
                if condition is not UNKNOWN:
                    self.statements(node.body if condition else node.orelse)
                else:
                    # Do not speculate through dynamic guards. Preserve dependencies of assignments,
                    # but explicitly mark module calls reached on either possible branch.
                    before, start = dict(self.env), len(self.flow.events)
                    self.statements(node.body)
                    left = dict(self.env)
                    self.env = dict(before)
                    self.statements(node.orelse)
                    for name in left.keys() | self.env.keys():
                        a, b = left.get(name, StaticValue()), self.env.get(name, StaticValue())
                        self.env[name] = a if a == b else combine([a, b])
                    for event in self.flow.events[start:]:
                        event.metadata["conditional"] = ast.unparse(node.test)
                    if len(self.flow.events) > start:
                        self.flow.warnings.append(
                            f"Conditional source path is unresolved: {ast.unparse(node.test)}"
                        )
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
                self.statements(node.body)

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

    def expression(self, node: ast.AST | None) -> StaticValue:
        if node is None:
            return StaticValue(constant=None)
        if isinstance(node, ast.Constant):
            return StaticValue(constant=node.value)
        if isinstance(node, ast.Name):
            return self.env.get(node.id, StaticValue())
        if isinstance(node, ast.Attribute):
            return self.attribute(self.expression(node.value), node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.expression(item) for item in node.elts]
            result = combine(values)
            result.items = values
            return result
        if isinstance(node, ast.Subscript):
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
            return (
                self.expression(node.body if condition else node.orelse)
                if condition is not UNKNOWN
                else combine([self.expression(node.body), self.expression(node.orelse)])
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
                return receiver
            if name in {"enumerate", "iter"} and node.args:
                return self.expression(node.args[0])
            arguments = {str(i): self.expression(arg) for i, arg in enumerate(node.args)}
            arguments.update(
                {kw.arg: self.expression(kw.value) for kw in node.keywords if kw.arg is not None}
            )
            if function.module_path is not None and function.method is None:
                return self.event(node, function.module_path, "module", function.module_path, arguments)
            if function.method is not None and self.depth < 3:
                return self.helper(function.module_path, function.method, list(arguments.values()), node)
            if name == "apply_chunking_to_forward" and node.args:
                helper = arguments["0"]
                if helper.method:
                    return self.helper(helper.module_path, helper.method, list(arguments.values())[3:], node)
            if name.startswith("create_") and name.endswith("mask"):
                return StaticValue()
            if name == "arange":
                return self.input("position_ids")
            if name in {"matmul", "bmm", "einsum", "cat", "stack"}:
                kind = "matmul" if name in {"matmul", "bmm", "einsum"} else "concatenation"
                return self.event(
                    node, None, kind, "Matrix multiply" if kind == "matmul" else "Concatenate", arguments
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
            return result
        return StaticValue()

    def helper(
        self, path: str | None, method_name: str, values: list[StaticValue], node: ast.AST
    ) -> StaticValue:
        module = self.model.get_submodule(path) if path else self.model
        try:
            method = getattr(type(module), method_name)
            function = next(
                n
                for n in ast.parse(textwrap.dedent(inspect.getsource(method))).body
                if isinstance(n, ast.FunctionDef)
            )
        except (AttributeError, OSError, TypeError, SyntaxError, StopIteration):
            self.flow.warnings.append(f"Helper source unavailable: {path}.{method_name}; dependencies only")
            return combine(values)
        old_env, old_output = self.env, self.flow.outputs
        self.env = {"self": StaticValue(module_path=path)}
        self.env.update({arg.arg: value for arg, value in zip(function.args.args[1:], values)})
        self.depth += 1
        self.statements(function.body)
        result = self.flow.outputs
        self.depth -= 1
        self.env, self.flow.outputs = old_env, old_output
        return result
