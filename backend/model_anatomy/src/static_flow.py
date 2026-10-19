"""A bounded AST dependency interpreter that never evaluates model computations."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import math
import operator as arithmetic
import textwrap
import dataclasses
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any

import torch
from transformers.utils.import_utils import is_torch_fx_proxy, is_torchdynamo_compiling

from .inputs import InputResolver
from .ir import TensorSpec
from .shape_rules import broadcast, selection as indexed_shape, transform, matmul, concatenate, module_contract
from .source_eval import DTYPES, TORCH_PRIMITIVES, FloatInfo, ModuleState, SourceBoundary, SourceEvaluator, Symbol, Tensor

UNKNOWN = object()


def argument_defaults(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr]:
    """Return declared defaults by AST parameter name, including keyword-only parameters."""
    args = function.args
    positional = args.posonlyargs + args.args
    defaults = dict(zip([arg.arg for arg in positional][-len(args.defaults):], args.defaults)) if args.defaults else {}
    defaults.update({arg.arg: value for arg, value in zip(args.kwonlyargs, args.kw_defaults) if value is not None})
    return defaults


def literal_constant(node: ast.AST) -> Any:
    """Read small literal defaults without executing publisher expressions or globals."""
    if sum(1 for _ in ast.walk(node)) > 256:
        return UNKNOWN
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return UNKNOWN

    def bounded(item: Any, depth: int = 0) -> bool:
        if depth > 8:
            return False
        if type(item) in (type(None), bool):
            return True
        if type(item) is int:
            return item.bit_length() <= 64
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is str:
            return len(item) <= 4096
        if type(item) in (list, tuple):
            return len(item) <= 128 and all(bounded(part, depth + 1) for part in item)
        if type(item) is dict:
            return len(item) <= 128 and all(type(key) is str and bounded(part, depth + 1) for key, part in item.items())
        return False

    return value if bounded(value) else UNKNOWN


@dataclass
class StaticValue:
    refs: list[tuple[str, str]] = field(default_factory=list)
    constant: Any = UNKNOWN
    module_path: str | None = None
    method: str | None = None
    items: list[StaticValue] | None = None
    fields: dict[str, StaticValue] | None = None
    may_be_none: bool = False
    unresolved_call: dict[str, Any] | None = None
    domain: str = "unknown"
    symbolic: Tensor | None = None
    shape_spec: TensorSpec | None = None
    index_bounds: tuple[int, int] | None = None

    @classmethod
    def literal(cls, value: Any) -> StaticValue:
        result = cls(constant=value, domain="scalar")
        if type(value) in (list, tuple):
            result.domain = "container"
            result.items = [cls.literal(item) for item in value]
        elif type(value) is dict:
            result.domain = "container"
            result.fields = {key: cls.literal(item) for key, item in value.items()}
        return result


@dataclass
class FlowEvent:
    id: str
    module_path: str | None
    kind: str
    label: str
    arguments: dict[str, StaticValue]
    line: int
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved_graph: Any = None


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
    resolved_operations: list[dict[str, Any]] = field(default_factory=list)
    unknown_inputs: set[str] = field(default_factory=set)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    wrapper_proof: dict[str, Any] | None = None


def combine(values: list[StaticValue]) -> StaticValue:
    return StaticValue(
        list(dict.fromkeys(ref for value in values for ref in value.refs)),
        may_be_none=any(value.may_be_none or value.constant is None for value in values),
    )


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

    def __init__(self, model: torch.nn.Module, module_path: str, classifications: dict[str, Any],
                 module_resolver: Any = None, argument_overrides: dict[str, StaticValue] | None = None):
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
        self.forward_parameters: dict[type, tuple[list[str], dict[str, Any]] | None] = {}
        self.helper_boundaries: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.speculation_depth = 0
        self.module_resolver = module_resolver
        self.argument_overrides = argument_overrides or {}

    def analyze(self) -> StaticFlow:
        if ("forward" in vars(self.module) or self.module._forward_hooks or self.module._forward_pre_hooks
                or inspect.getattr_static(type(self.module), "__getattribute__", object.__getattribute__) is not object.__getattribute__
                or inspect.getattr_static(type(self.module), "__getattr__", torch.nn.Module.__getattr__) is not torch.nn.Module.__getattr__):
            value = self.unsupported(ast.Pass(lineno=0, col_offset=0),
                "Instance forward overrides, hooks or custom attribute lookup require separate source proof")
            self.flow.outputs = value
            return self.flow
        from .wrapper_proof import forward_wrappers
        wrappers = forward_wrappers(type(self.module).forward, self.module,
                                    {name: value.constant for name, value in self.argument_overrides.items()})
        self.flow.wrapper_proof = wrappers
        if wrappers["status"] == "unknown":
            self.flow.outputs = self.unsupported(ast.Pass(lineno=0, col_offset=0),
                "Forward decorator effects are unproven: " + "; ".join(wrappers["failure_reasons"]))
            return self.flow
        try:
            method = inspect.unwrap(type(self.module).forward)
            self.source_globals = method.__globals__
            source = textwrap.dedent(inspect.getsource(method))
            self.source_name = f"{method.__module__}.{method.__qualname__}"
            self.source_hash = hashlib.sha256(source.encode()).hexdigest()
            tree = ast.parse(source)
            function = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        except (OSError, TypeError, AttributeError, SyntaxError, StopIteration) as exc:
            self.flow.warnings.append(f"Forward source unavailable: {type(exc).__name__}")
            return self.flow
        self.collect_source_calls(function)
        defaults = argument_defaults(function)
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
            basis = "Canonical static profile; not a runtime observation"
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
                value = literal_constant(defaults[name]) if name in defaults else UNKNOWN
                basis = "Declared source default"
                if value is UNKNOWN:
                    self.flow.unknown_inputs.add(name)
                    self.env[name] = self.input(name, may_be_none=True)
                else:
                    self.env[name] = StaticValue.literal(value)
            if name in self.argument_overrides:
                supplied = self.argument_overrides[name]
                self.env[name] = self.local_argument(name, supplied)
                basis = "Caller argument propagated from the source graph; not a runtime observation"
                if supplied.symbolic is not None or supplied.constant is not UNKNOWN:
                    self.flow.unknown_inputs.discard(name)
            value = self.env[name]
            if type(value.constant) in (type(None), bool, int, float, str, list, tuple, dict) and not isinstance(value.constant, Symbol):
                try:
                    profile_value = json.loads(json.dumps(value.constant, allow_nan=False))
                except (TypeError, ValueError):
                    self.flow.input_profile.append({"argument": name, "known": False,
                        "basis": "Symbolic metadata is not a concrete profile literal", **self.source_location(argument)})
                    continue
                self.flow.input_profile.append({
                    "argument": name, "value": profile_value,
                    "value_type": type(value.constant).__name__,
                    "basis": basis, "known": True,
                    **self.source_location(argument),
                })
            elif name in self.flow.unknown_inputs:
                self.flow.input_profile.append({
                    "argument": name, "known": False,
                    "basis": "Unresolved source default" if name in defaults else "Required argument absent from canonical profile",
                    **({"default_expression": ast.unparse(defaults[name])} if name in defaults else {}),
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
            declared = {arg.arg for arg in function.args.posonlyargs + function.args.args + function.args.kwonlyargs}
            extras = {name: self.local_argument(f"{function.args.kwarg.arg}.{name}", value)
                      for name, value in self.argument_overrides.items() if name not in declared}
            value = combine(list(extras.values()))
            value.fields, value.domain = extras, "container"
            self.env[function.args.kwarg.arg] = value
            self.flow.input_profile.append({
                "argument": function.args.kwarg.arg, "argument_names": sorted(extras),
                "basis": "Actual caller extra keywords" if extras else "Canonical profile supplies no additional keyword arguments",
                **self.source_location(function.args.kwarg),
            })
        self.statements(function.body)
        self.flow.outputs = merge_outputs(self.returned)
        return self.flow

    def local_argument(self, name: str, supplied: StaticValue) -> StaticValue:
        if supplied.constant is not UNKNOWN:
            return supplied
        if supplied.fields is not None:
            fields = {key: self.local_argument(f"{name}.{key}", value) for key, value in supplied.fields.items()}
            value = combine(list(fields.values()))
            value.fields, value.domain = fields, supplied.domain
            value.items = [item for item in fields.values() if item.constant is not None]
            return value
        if supplied.items is not None:
            items = [self.local_argument(f"{name}.{i}", item) for i, item in enumerate(supplied.items)]
            value = combine(items)
            value.items, value.domain = items, "tuple"
            return value
        local = self.input(name, may_be_none=supplied.may_be_none)
        local.symbolic, local.shape_spec, local.domain = supplied.symbolic, supplied.shape_spec, supplied.domain
        self.flow.inputs[name] = local
        return local

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
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            label = f"Unresolved call: {name}"
            detail = self.call_detail(node, reason)
            key = (self.source_name, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            if key in self.helper_boundaries:
                detail["helper_boundary"] = self.helper_boundaries[key]
        elif isinstance(node, ast.stmt):
            label = f"Unsupported statement: {type(node).__name__}"
            detail["category"] = "statement"
        else:
            label = f"Unresolved expression: {ast.unparse(node)}"
            detail["category"] = "expression"
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
        value = self.event(node, None, "unknown", label[:160],
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
        if isinstance(value, Symbol):
            value = UNKNOWN
        evidence = {"condition": ast.unparse(node),
                    "resolved": value is not UNKNOWN,
                    "value": bool(value) if value is not UNKNOWN else None,
                    "basis": "Static attributes and canonical input profile",
                    **self.source_location(node)}
        self.flow.branch_conditions.append(evidence)
        return evidence

    def input(self, name: str, *, may_be_none: bool = False) -> StaticValue:
        value = StaticValue([(f"input:{name}", "output")], may_be_none=may_be_none)
        if not may_be_none:
            spec = InputResolver(self.model).spec(name, self.path)
            value.shape_spec, value.domain = spec, "tensor"
            if spec.shape is not None and spec.dtype is not None:
                value.symbolic = Tensor(tuple(Symbol(d) if isinstance(d, str) else d for d in spec.shape),
                                        spec.dtype, name)
                value.shape_spec, value.domain = spec, "tensor"
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
        event = FlowEvent(identifier, path, kind, label, arguments, getattr(node, "lineno", 0), metadata or {})
        event.metadata["source_location"] = self.source_location(node)
        self.flow.events.append(event)
        value = StaticValue([(identifier, "output")], may_be_none=kind == "unknown")
        if path is not None and self.module_resolver is not None:
            return self.module_resolver(event, arguments)
        if path is not None:
            module = self.model.get_submodule(path) if path else self.model
            spec = next((self.tensor_spec(item) for item in arguments.values() if item.domain == "tensor"), None)
            check = module_contract(module, spec)
            if check is not None:
                return self.attach_shape(value, event, check)
            incoming = next((v.symbolic for v in arguments.values() if v.symbolic is not None), None)
            if type(module) is torch.nn.Embedding and "forward" not in vars(module) and incoming is not None:
                dtype = DTYPES.get(module.weight.dtype)
                if dtype:
                    value.symbolic = Tensor((*incoming.shape, module.embedding_dim), dtype, f"embedding({incoming.expression})")
                    value.domain = "tensor"
                    value.shape_spec = TensorSpec(list(value.symbolic.spec()["shape"]), dtype, "embedding")
                    self.resolved(ast.Constant(value=None), "native_embedding_output", module_path=path,
                                  weight_shape=list(module.weight.shape), input=incoming.spec(), output=value.symbolic.spec())
        return value

    def assign(self, target: ast.AST, value: StaticValue) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if type(value.constant) in (tuple, list):
                value = replace(value, items=[StaticValue.literal(item) for item in value.constant])
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
            elif isinstance(node, ast.Raise):
                if self.speculation_depth == 0:
                    from .errors import UnsupportedModelError

                    raise UnsupportedModelError(f"Recorded static input profile reaches {ast.unparse(node)} in {self.source_name}:{node.lineno}; no exception constructor was executed")
                self.unsupported(node, "Conditional exception path remains possible; exception was not executed")
                return True
            elif isinstance(node, ast.Expr):
                self.expression(node.value)
            elif isinstance(node, ast.If):
                condition = self.expression(node.test).constant
                if isinstance(condition, Symbol):
                    condition = UNKNOWN
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
                    left_returns = self.speculative_statements(node.body)
                    left = dict(self.env)
                    for event in self.flow.events[start:]:
                        self.mark_conditional(event, ast.unparse(node.test))
                        event.metadata.setdefault("branch_evidence", []).append(dict(evidence, arm="body"))
                    right_start = len(self.flow.events)
                    self.env = dict(before)
                    right_returns = self.speculative_statements(node.orelse)
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
                        returned = self.speculative_statements(statements[index + 1 :])
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
                    if (isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2
                            and isinstance(node.iter, ast.Call) and self.global_symbol(node.iter.func) is enumerate
                            and len(node.iter.args) == 1 and not node.iter.keywords):
                        container = self.model.get_submodule(iterable.module_path)
                        self.assign(node.target.elts[0], StaticValue(domain="loop_index",
                                    index_bounds=(0, len(container._modules))))
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
                if isinstance(node, ast.With) and len(node.items) == 1 and node.items[0].optional_vars is None:
                    context = node.items[0].context_expr
                    if isinstance(context, ast.Call) and self.global_symbol(context.func) is torch.autocast:
                        options = {keyword.arg: self.expression(keyword.value) for keyword in context.keywords}
                        args = [self.expression(arg) for arg in context.args]
                        if (options.get("enabled", StaticValue()).constant is False and len(args) <= 1
                                and options.keys() <= {"enabled", "device_type", "dtype", "cache_enabled"}
                                and not any(value.refs for value in [*args, *options.values()])):
                            self.resolved(context, "disabled_native_autocast", enabled=False,
                                          condition="Valid native device_type; autocast disabled in this source scope")
                            if self.statements(node.body):
                                return True
                            continue
                self.unsupported(node, "Context manager entry/exit and exception effects are not established")
            elif isinstance(node, (ast.Try, ast.TryStar)):
                self.unsupported(node, "Try body, handlers, else and finally remain possible; exception flow is unsupported")
            else:
                self.unsupported(node, "Statement semantics are not supported by the static interpreter")
        return False

    def speculative_statements(self, nodes: list[ast.stmt]) -> bool:
        self.speculation_depth += 1
        try:
            return self.statements(nodes)
        finally:
            self.speculation_depth -= 1

    @staticmethod
    def mark_conditional(event: FlowEvent, condition: str) -> None:
        existing = event.metadata.get("conditional")
        event.metadata["conditional"] = f"({existing}) and ({condition})" if existing else condition

    def operation(
        self, node: ast.AST, left: StaticValue, right: StaticValue, operator: ast.operator
    ) -> StaticValue:
        if (isinstance(operator, ast.Add) and type(left.constant) in (list, tuple)
                and type(left.constant) is type(right.constant) and len(left.constant) + len(right.constant) <= 128):
            value = StaticValue.literal(left.constant + right.constant)
            if left.domain == "shape" or right.domain == "shape":
                value.domain = "shape"
            self.resolved(node, "literal_sequence_concatenation", item_count=len(value.constant))
            return value
        if isinstance(operator, ast.Mult):
            sequence, count = (left, right) if left.items is not None else (right, left)
            if sequence.items is not None and type(count.constant) is int and 0 <= count.constant <= 128 and len(sequence.items) * count.constant <= 128:
                items = sequence.items * count.constant
                value = combine(items)
                value.items, value.domain = items, "tuple"
                if all(item.constant is not UNKNOWN for item in items):
                    value.constant = tuple(item.constant for item in items)
                self.resolved(node, "bounded_sequence_repetition", item_count=len(items))
                return value
        if isinstance(operator, ast.Add) and left.items is not None and right.items is not None:
            items = left.items + right.items
            value = combine(items)
            value.items, value.domain = items, "tuple"
            if all(item.constant is not UNKNOWN for item in items):
                value.constant = tuple(item.constant for item in items)
            self.resolved(node, "container_concatenation", item_count=len(items))
            return value
        if type(left.constant) is bool and type(right.constant) is bool and isinstance(operator, (ast.BitXor, ast.BitAnd, ast.BitOr)):
            operations = {ast.BitXor: arithmetic.xor, ast.BitAnd: arithmetic.and_, ast.BitOr: arithmetic.or_}
            result = operations[type(operator)](left.constant, right.constant)
            self.resolved(node, "boolean_bitwise", operator=type(operator).__name__, operands=[left.constant, right.constant], value=result)
            return StaticValue.literal(result)
        if isinstance(left.constant, Symbol) or isinstance(right.constant, Symbol):
            if isinstance(operator, ast.Add) and type(left.constant) is int and left.constant == 0:
                return right
            if isinstance(operator, ast.Add) and type(right.constant) is int and right.constant == 0:
                return left
            try:
                return StaticValue(constant=SourceEvaluator().binary(left.constant, right.constant, operator))
            except SourceBoundary:
                return StaticValue()
        if not left.refs and not right.refs:
            operations = {
                ast.Add: arithmetic.add,
                ast.Sub: arithmetic.sub,
                ast.Mult: arithmetic.mul,
                ast.Div: arithmetic.truediv,
                ast.FloorDiv: arithmetic.floordiv,
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
            if left.unresolved_call or right.unresolved_call:
                dependencies = [self.index_dependencies(left), self.index_dependencies(right)]
                return self.unsupported(node, "Scalar arithmetic depends on unresolved calls", dependencies)
            return StaticValue()
        if not left.refs or not right.refs:
            tensor, scalar = (left, right) if left.refs else (right, left)
            spec = self.tensor_spec(tensor)
            if isinstance(operator, ast.MatMult):
                value = self.event(node, None, "matmul", "Matrix multiply", {"input": tensor})
                check = matmul(spec, TensorSpec([], None, "scalar"))
                check.status, check.reason = "invalid", "Matmul operands must be tensors of rank at least one"
                return self.attach_shape(value, self.flow.events[-1], check)
            if (tensor.domain == "tensor" and type(scalar.constant) in (int, float)
                    and isinstance(operator, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow))):
                check = transform(spec, "contiguous", [])
                check.rule = "tensor_scalar_" + type(operator).__name__
                if (spec.dtype not in {"F16", "BF16", "F32", "F64"}
                        or isinstance(operator, ast.Div) and right.constant == 0):
                    check.spec = TensorSpec(None, None, "unresolved_scalar_promotion")
                    check.status, check.reason = "unknown", "Scalar promotion or division domain is unproven"
                value = self.event(node, None, "scale" if isinstance(operator, (ast.Mult, ast.Div)) else "operation",
                                   "Tensor scalar operation", {"input": tensor},
                                   {"operator": type(operator).__name__, "scalar": scalar.constant,
                                    "scalar_side": "right" if tensor is left else "left"})
                return self.attach_shape(value, self.flow.events[-1], check)
            return self.unsupported(node, "Tensor arithmetic operand or scalar value is unproven", [left, right])
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
        value = self.event(node, None, kind, label, {"left": left, "right": right}, metadata)
        check = (matmul(self.tensor_spec(left), self.tensor_spec(right)) if isinstance(operator, ast.MatMult)
                 else broadcast([self.tensor_spec(left), self.tensor_spec(right)], "elementwise_" + type(operator).__name__))
        self.attach_shape(value, self.flow.events[-1], check)
        return value

    @staticmethod
    def tensor_spec(value: StaticValue) -> TensorSpec:
        if value.symbolic is not None:
            spec = value.symbolic.spec()
            return TensorSpec(spec["shape"], spec["dtype"], "symbolic_tensor")
        return value.shape_spec or TensorSpec(None, None, "unknown")

    @staticmethod
    def attach_shape(value: StaticValue, event: FlowEvent, check: Any) -> StaticValue:
        value.domain, value.shape_spec = "tensor", check.spec
        event.metadata["shape_check"] = check.record()
        event.metadata["proven_output"] = {"shape": check.spec.shape, "dtype": check.spec.dtype}
        if check.spec.shape is not None and check.spec.dtype is not None:
            value.symbolic = Tensor(tuple(Symbol(d) if isinstance(d, str) else d for d in check.spec.shape),
                                    check.spec.dtype, event.label)
        return value

    def attribute(self, value: StaticValue, name: str) -> StaticValue:
        if isinstance(value.constant, Symbol) and value.constant.expression == "input_device" and name == "type":
            return StaticValue(constant=Symbol("input_device_type"), domain="device_type")
        if value.fields is not None and name in value.fields:
            return value.fields[name]
        if isinstance(value.constant, FloatInfo):
            return StaticValue(constant=SourceEvaluator().attribute(value.constant, name))
        if value.symbolic is not None and name in {"shape", "dtype", "device", "ndim"}:
            constant = SourceEvaluator().attribute(value.symbolic, name)
            return StaticValue(constant=constant, domain="shape" if name == "shape" else "metadata")
        if name in {"shape", "dtype", "device", "ndim"}:
            return StaticValue(domain="shape" if name == "shape" else "metadata")
        if value.module_path is not None:
            path = f"{value.module_path}.{name}" if value.module_path else name
            if path in self.tainted_attributes:
                return self.tainted_attributes[path]
            if path in self.classifications:
                return StaticValue(module_path=path)
            module = self.model.get_submodule(value.module_path) if value.module_path else self.model
            attribute = next((registry[name] for registry in (module._parameters, module._buffers) if name in registry), UNKNOWN)
            if attribute is UNKNOWN:
                attribute = vars(module).get(name, inspect.getattr_static(module, name, UNKNOWN))
            if isinstance(attribute, property):
                state = self.module_state(module)
                return StaticValue(constant=state.attributes[name], domain="metadata") if name in state.attributes else StaticValue()
            if isinstance(attribute, torch.Tensor):
                buffer = name in module._buffers
                dtype = DTYPES.get(attribute.dtype)
                spec = TensorSpec(list(attribute.shape), dtype, "native_buffer" if buffer else "native_parameter")
                return StaticValue([(f"{'buffer' if buffer else 'parameter'}:{path}", "output")],
                                   domain="tensor", shape_spec=spec,
                                   symbolic=Tensor(tuple(attribute.shape), dtype, path) if dtype else None)
            if inspect.ismethod(attribute) or inspect.isfunction(attribute):
                return StaticValue(module_path=value.module_path, method=name)
            return StaticValue(constant=attribute)
        if value.constant is not UNKNOWN and value.constant is not None:
            if isinstance(value.constant, ModuleType):
                return StaticValue(constant=vars(value.constant).get(name, UNKNOWN))
            try:
                return StaticValue(constant=SourceEvaluator().attribute(value.constant, name))
            except (SourceBoundary, AttributeError, TypeError, ValueError):
                return StaticValue()
        return StaticValue([(ref, name) for ref, _ in value.refs])

    def global_symbol(self, expression: ast.AST) -> Any:
        """Resolve globals through plain module dictionaries without invoking attributes."""
        parts = []
        while isinstance(expression, ast.Attribute):
            parts.append(expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name) or expression.id in self.env:
            return UNKNOWN
        function = self.source_globals.get(expression.id, vars(builtins).get(expression.id, UNKNOWN))
        for name in reversed(parts):
            if not isinstance(function, ModuleType):
                return UNKNOWN
            function = vars(function).get(name, UNKNOWN)
        return function

    def resolved(self, node: ast.AST, rule: str, **details: Any) -> None:
        self.flow.resolved_operations.append({
            "rule": rule, "expression": ast.unparse(node),
            **self.source_location(node), **details,
        })

    def call_detail(self, node: ast.Call, reason: str) -> dict[str, Any]:
        detail = {"construct": "Call", "expression": ast.unparse(node),
                  "reason": reason, "category": "call", "call_target": ast.unparse(node.func),
                  **self.source_location(node)}
        function = self.global_symbol(node.func)
        if inspect.isfunction(function) or inspect.isbuiltin(function) or function is builtins.slice:
            try:
                source_hash = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            except (OSError, TypeError):
                source_hash = None
            detail["callable"] = {
                "qualified_name": f"{function.__module__}.{function.__qualname__}",
                "source_sha256": source_hash,
            }
        return detail

    def deferred_unknown_call(self, node: ast.Call) -> StaticValue:
        detail = self.call_detail(node, "Opaque call without known tensor dependencies; computation is unverified")
        self.flow.unexplained_operations.append(detail)
        self.flow.warnings.append(f"Opaque call {detail['call_target']}: result type and computation are unverified")
        return StaticValue(may_be_none=True, unresolved_call=detail)

    def index_dependencies(self, value: StaticValue) -> StaticValue:
        if value.unresolved_call and not value.refs:
            detail = value.unresolved_call
            location = ast.Pass(lineno=detail["line"], col_offset=detail["column"])
            result = self.event(location, None, "unknown", f"Unresolved call: {detail['call_target']}",
                                {}, {"unsupported": detail, "usage_roles": ["indices"]})
            value.refs = result.refs
        nested = [self.index_dependencies(item) for item in (value.items or [])]
        return combine([value, *nested])

    def module_argument_roles(self, path: str, arguments: dict[str, StaticValue]) -> dict[str, Any]:
        """Name positional ports only when the native forward signatures agree."""
        module = self.model.get_submodule(path) if path else self.model
        targets = list(module.children()) if isinstance(module, torch.nn.ModuleList) else [module]
        signatures = []
        sources = {}
        for target_type in dict.fromkeys(type(target) for target in targets):
            if target_type not in self.forward_parameters:
                try:
                    method = inspect.unwrap(target_type.forward)
                    source = textwrap.dedent(inspect.getsource(method))
                    function = next(item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef))
                    names = [arg.arg for arg in function.args.posonlyargs + function.args.args]
                    if names and names[0] == "self":
                        names = names[1:]
                    self.forward_parameters[target_type] = (names, {
                        "qualified_name": f"{method.__module__}.{method.__qualname__}",
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    })
                except (AttributeError, OSError, TypeError, SyntaxError, StopIteration):
                    self.forward_parameters[target_type] = None
            signature = self.forward_parameters[target_type]
            if signature is None:
                signatures = []
                break
            names, evidence = signature
            signatures.append(names)
            sources[evidence["qualified_name"]] = evidence
        roles = {name: name for name in arguments if not name.isdigit()}
        if signatures and all(names == signatures[0] for names in signatures):
            roles.update({name: signatures[0][int(name)] for name in arguments
                          if name.isdigit() and int(name) < len(signatures[0])})
        return {"argument_roles": roles, "argument_role_sources": list(sources.values())}

    def selection(self, node: ast.Subscript, value: StaticValue, index: StaticValue) -> StaticValue:
        dependencies = self.index_dependencies(index)
        indices = index.items if isinstance(node.slice, ast.Tuple) and index.items is not None else [index]
        descriptors = []
        full_slice = True
        for item in indices:
            constant = item.constant
            if type(constant) is slice:
                descriptors.append({"kind": "slice", "start": constant.start.expression if isinstance(constant.start, Symbol) else constant.start,
                                    "stop": constant.stop.expression if isinstance(constant.stop, Symbol) else constant.stop,
                                    "step": constant.step.expression if isinstance(constant.step, Symbol) else constant.step})
                full_slice = full_slice and constant.start in (None, 0) and constant.stop is None and constant.step in (None, 1)
            elif type(constant) in (int, type(None)):
                descriptors.append({"kind": "integer" if type(constant) is int else "new_axis", "value": constant})
                full_slice = False
            elif constant is Ellipsis:
                descriptors.append({"kind": "ellipsis"})
            else:
                descriptors.append({"kind": "unknown"})
                full_slice = False
        rank = self.tensor_spec(value).shape
        if rank is not None and sum(item["kind"] == "ellipsis" for item in descriptors) == 1:
            consumed = sum(item["kind"] not in {"ellipsis", "new_axis"} for item in descriptors)
            if consumed <= len(rank):
                descriptors = [part for item in descriptors for part in
                    ([{"kind": "slice", "start": None, "stop": None, "step": None}] * (len(rank) - consumed)
                     if item["kind"] == "ellipsis" else [item])]
        result = self.event(node, None, "selection", "Select tensor elements",
                          {"input": value, "indices": dependencies}, {"indices": [
                              None if item.get("kind") == "slice" and all(item.get(key) is None for key in ("start", "stop", "step"))
                              else item["value"] if item.get("kind") == "integer" else "unknown" for item in descriptors], "indexing": {
                              "indices": descriptors, "full_slice": full_slice and not dependencies.refs,
                              "expression": ast.unparse(node),
                              "shape_proven": False, **self.source_location(node),
                          }})
        spec = self.tensor_spec(value)
        check = indexed_shape(spec, descriptors)
        self.attach_shape(result, self.flow.events[-1], check)
        if full_slice and not dependencies.refs and check.status == "proven" and spec.shape is not None:
            self.flow.events[-1].metadata["indexing"].update(shape_proven=True, rule="full_slice_identity", input_rank=len(spec.shape))
            self.resolved(node, "full_slice_identity", input_rank=len(spec.shape))
        return result

    def canonical_slice(self, node: ast.Call, arguments: dict[str, StaticValue]) -> StaticValue | None:
        if self.global_symbol(node.func) is not builtins.slice or node.keywords or not 1 <= len(node.args) <= 3:
            return None
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return None
        values = [arguments[str(index)] for index in range(len(node.args))]
        if any(value.refs or type(value.constant) not in (int, type(None)) for value in values):
            return None
        descriptor = builtins.slice(*(value.constant for value in values))
        if descriptor.step == 0:
            return self.unsupported(node, "Slice step cannot be zero", values)
        self.resolved(node, "builtin_slice", result={
            "start": descriptor.start, "stop": descriptor.stop, "step": descriptor.step,
        })
        return StaticValue(constant=descriptor)

    def canonical_predicate(self, node: ast.Call, arguments: dict[str, StaticValue]) -> StaticValue | None:
        """Resolve native predicates by identity, never by the spelling of a call."""
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return None
        function = self.global_symbol(node.func)
        if function is isinstance:
            if len(node.args) != 2 or node.keywords:
                return None
            target_nodes = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
            targets = tuple(self.global_symbol(target) for target in target_nodes)
            allowed = (dict, list, tuple, int, bool, float, str)
            if not targets or not all(any(target is candidate for candidate in allowed) for target in targets):
                return None
            value = arguments["0"].constant
            if type(value) in (type(None), bool, int, float, str, dict, list, tuple):
                result = isinstance(value, targets)
                self.resolved(node, "builtin_isinstance", value=result,
                              target_types=[target.__name__ for target in targets])
                return StaticValue(constant=result)
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
            if node.id in self.env:
                return self.env[node.id]
            constant = self.global_symbol(node)
            if inspect.isfunction(constant) or isinstance(constant, ModuleType) or any(constant is item for item in (str, int, float, bool, tuple, list)) or type(constant) in (bool, int, float, str, type(None), torch.dtype):
                return StaticValue(constant=constant, domain="callable" if inspect.isfunction(constant) else "metadata")
            return StaticValue()
        if isinstance(node, ast.NamedExpr):
            value = self.expression(node.value)
            self.assign(node.target, value)
            return value
        if isinstance(node, ast.Attribute):
            constant = self.global_symbol(node)
            if type(constant) is torch.dtype or type(constant) in (bool, int, float, str, type(None)):
                return StaticValue(constant=constant, domain="metadata")
            return self.attribute(self.expression(node.value), node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            values = []
            for item in node.elts:
                value = self.expression(item.value if isinstance(item, ast.Starred) else item)
                if isinstance(item, ast.Starred):
                    expanded = value.items if value.items is not None else [StaticValue.literal(part) for part in value.constant] if type(value.constant) in (list, tuple) else None
                    if expanded is None or len(expanded) + len(values) > 128:
                        return self.unsupported(node, "Tuple unpacking dimensions are unproven", [value])
                    values.extend(expanded)
                else:
                    values.append(value)
            result = combine(values)
            result.items = values
            result.domain = "tuple"
            if all(item.constant is not UNKNOWN for item in values):
                result.constant = tuple(item.constant for item in values)
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
            bounds = [self.expression(part) for part in (node.lower, node.upper, node.step)]
            if all(not value.refs and (type(value.constant) in (int, type(None)) or isinstance(value.constant, Symbol)) for value in bounds):
                descriptor = slice(*(value.constant for value in bounds))
                if descriptor.step == 0:
                    return self.unsupported(node, "Slice step cannot be zero", bounds)
                return StaticValue(constant=descriptor)
            return combine(bounds)
        if isinstance(node, ast.Subscript):
            value, index_value = self.expression(node.value), self.expression(node.slice)
            index = index_value.constant
            if value.domain == "shape":
                if type(value.constant) is tuple and type(index) in (int, slice):
                    try:
                        selected = value.constant[index]
                        self.resolved(node, "shape_metadata_index", result_domain="metadata" if type(index) is int else "shape")
                        return StaticValue(constant=selected, domain="metadata" if type(index) is int else "shape")
                    except (IndexError, TypeError):
                        pass
                result = combine([value, index_value])
                result.domain = "metadata" if type(index) is int else "shape"
                self.resolved(node, "shape_metadata_index", result_domain=result.domain)
                return result
            if type(value.constant) in (list, tuple, str, dict) and type(index) in (int, str, slice):
                try:
                    selected = value.constant[index]
                except (IndexError, KeyError, TypeError, ValueError):
                    return self.unsupported(node, "Literal indexing is invalid in the recorded profile", [value, index_value])
                self.resolved(node, "literal_index", result_type=type(selected).__name__)
                return StaticValue.literal(selected)
            if (type(value.constant) in (list, tuple) and value.constant
                    and index_value.index_bounds is not None
                    and index_value.index_bounds[0] >= 0 and index_value.index_bounds[1] <= len(value.constant)
                    and all(type(item) in (type(None), bool, int, float, str) and item == value.constant[0] for item in value.constant)):
                self.resolved(node, "homogeneous_metadata_index", index_bounds=list(index_value.index_bounds),
                              sequence_length=len(value.constant), value=value.constant[0])
                return StaticValue.literal(value.constant[0])
            if value.module_path is not None and type(index) is int:
                module = self.model.get_submodule(value.module_path) if value.module_path else self.model
                names = list(module._modules)
                if -len(names) <= index < len(names):
                    return StaticValue(module_path=value.module_path + "." + names[index])
            if value.module_path is not None:
                return value
            if value.items and type(index) is int and -len(value.items) <= index < len(value.items):
                return value.items[index]
            if value.items is not None and type(index) is slice:
                selected = value.items[index]
                result = combine(selected)
                result.items, result.domain = selected, "container"
                self.resolved(node, "source_sequence_slice", item_count=len(selected))
                return result
            if value.fields and type(index) is str and index in value.fields:
                return value.fields[index]
            if value.domain == "tensor" or isinstance(node.slice, (ast.Tuple, ast.Slice)) or type(index) is slice or index_value.refs:
                return self.selection(node, value, index_value)
            return StaticValue(
                [(ref, str(index) if index is not UNKNOWN else port) for ref, port in value.refs],
                may_be_none=value.may_be_none,
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.expression(node.operand)
            return (
                StaticValue(constant=not value.constant)
                if value.constant is not UNKNOWN and not isinstance(value.constant, Symbol) else StaticValue()
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand)
            if type(value.constant) in (int, float):
                return StaticValue(constant=-value.constant if isinstance(node.op, ast.USub) else value.constant)
            if value.domain == "tensor":
                result = self.event(node, None, "operation", "Negate" if isinstance(node.op, ast.USub) else "Identity", {"input": value})
                check = transform(self.tensor_spec(value), "contiguous", [])
                check.rule = "tensor_unary_sign"
                return self.attach_shape(result, self.flow.events[-1], check)
            return self.unsupported(node, "Unary operation is not statically established", [self.index_dependencies(value)])
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
            if type(left.constant) is torch.dtype and type(right.constant) is torch.dtype and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                same = left.constant is right.constant
                self.resolved(node, "native_dtype_comparison", left=str(left.constant), right=str(right.constant))
                return StaticValue.literal(same if isinstance(node.ops[0], ast.Eq) else not same)
            if isinstance(left.constant, Symbol) or isinstance(right.constant, Symbol):
                return StaticValue()
            if isinstance(node.ops[0], (ast.Is, ast.IsNot)) and right.constant is None:
                if left.refs and not left.may_be_none or left.module_path is not None:
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
            if isinstance(condition, Symbol):
                condition = UNKNOWN
            evidence = self.branch(node.test, condition)
            if condition is not UNKNOWN:
                self.exclude_source_calls([node.orelse if condition else node.body], evidence)
                start = len(self.flow.events)
                value = self.expression(node.body if condition else node.orelse)
                for event in self.flow.events[start:]:
                    event.metadata.setdefault("branch_evidence", []).append(evidence)
                return value
            before = dict(self.env)
            choice_events = len(self.flow.events)
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
            if left.refs or right.refs or len(self.flow.events) > choice_events or left_env != before or self.env != before:
                self.flow.warnings.append(f"Conditional source path is unresolved: {ast.unparse(node.test)}")
            else:
                self.resolved(node, "metadata_only_choice", condition_value_unknown=True)
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
            if isinstance(node.func, ast.Attribute) and name == "split" and receiver is not None:
                parts = [self.expression(arg).constant for arg in node.args]
                options = {kw.arg: self.expression(kw.value).constant for kw in node.keywords}
                spec = self.tensor_spec(receiver)
                axis = options.get("dim", parts[1] if len(parts) == 2 else 0)
                sections = parts[0] if parts else None
                if (spec.shape is not None and type(axis) is int and -len(spec.shape) <= axis < len(spec.shape)
                        and options.keys() <= {"dim"} and len(parts) in {1, 2}
                        and not (len(parts) == 2 and "dim" in options)):
                    axis %= len(spec.shape)
                    extent = spec.shape[axis]
                    sizes = list(sections) if type(sections) in (list, tuple) else [sections] * (extent // sections) if type(extent) is int and type(sections) is int and sections > 0 and extent % sections == 0 else None
                    if sizes is not None and len(sizes) <= 128 and all(type(size) is int and size >= 0 for size in sizes) and sum(sizes) == extent:
                        from .shape_values import bind, ports
                        structure = {"kind": "tuple", "items": [{"kind": "tensor", "shape": spec.shape[:axis] + [size] + spec.shape[axis + 1:], "dtype": spec.dtype} for size in sizes]}
                        value = self.event(node, None, "split", "Split tensor", {"input": receiver},
                            {"output_structure": structure, "output_ports": ports(structure), "source_rule": "native_tensor_split", "axis": axis, "sizes": sizes,
                             "shape_check": {"rule": "native_tensor_split", "status": "proven" if spec.dtype else "unknown",
                                             "reason": "" if spec.dtype else "Split dtype is unproven", "constraints": [], "runtime_verified": False}})
                        return bind(structure, value.refs[0][0])
                return self.unsupported(node, "Split sections, axis or input layout are unproven", [receiver])
            if (isinstance(node.func, ast.Attribute) and receiver is not None
                    and receiver.domain == "tensor" and name in {"pow", "mean", "cos", "sin"}):
                args = [self.expression(arg).constant for arg in node.args]
                options = {kw.arg: self.expression(kw.value).constant for kw in node.keywords}
                check = transform(self.tensor_spec(receiver), "contiguous", [])
                check.rule = "tensor_" + name
                if name == "mean":
                    axis = args[0] if args else options.get("dim")
                    shape = check.spec.shape
                    if (shape is not None and type(axis) is int and -len(shape) <= axis < len(shape)
                            and type(options.get("keepdim", False)) is bool
                            and set(options) <= {"dim", "keepdim"} and len(args) <= 1):
                        axis %= len(shape)
                        check.spec.shape = [1 if i == axis else d for i, d in enumerate(shape)] if options.get("keepdim") else [d for i, d in enumerate(shape) if i != axis]
                        if type(shape[axis]) is str:
                            check.constraints.append(f"{shape[axis]} > 0")
                            check.status = "conditional"
                    else:
                        check.spec.shape = None
                        check.status, check.reason = "unknown", "Reduction axes are unproven"
                elif name == "pow" and (len(args) != 1 or type(args[0]) not in (int, float) or options):
                    check.spec.shape = None
                    check.status, check.reason = "unknown", "Exponent is not an established scalar"
                elif name in {"cos", "sin"} and (args or options):
                    check.spec.shape = None
                    check.status, check.reason = "unknown", "Unary method arguments are unsupported"
                if name in {"mean", "cos", "sin"} and check.spec.dtype not in {"F16", "BF16", "F32", "F64"}:
                    check.spec = TensorSpec(None, None, "unproven_reduction_dtype")
                    check.status, check.reason = "unknown", "Reduction or unary dtype promotion is not established"
                if name == "pow" and check.spec.dtype not in {"F16", "BF16", "F32", "F64"}:
                    check.spec = TensorSpec(None, None, "unproven_power_dtype")
                    check.status, check.reason = "unknown", "Power dtype and exponent compatibility are not established"
                value = self.event(node, None, "source_operation", name.title(), {"input": receiver})
                return self.attach_shape(value, self.flow.events[-1], check)
            if isinstance(node.func, ast.Attribute) and receiver is not None and receiver.domain == "tensor" and name in {"size", "dim", "numel"}:
                if receiver.symbolic is not None and name in {"size", "dim"} and not node.keywords:
                    args = [self.expression(arg).constant for arg in node.args]
                    try:
                        constant = SourceEvaluator().method(receiver.symbolic, name, args, {}, node)
                        return StaticValue(constant=constant, domain="shape" if type(constant) is tuple else "metadata")
                    except (SourceBoundary, TypeError, IndexError):
                        pass
                return StaticValue(domain="shape" if name == "size" and not node.args and not node.keywords else "metadata")
            if isinstance(node.func, ast.Attribute) and receiver is not None and receiver.domain == "tensor" and name in {
                "view",
                "reshape",
                "contiguous",
                "to",
                "type",
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
                values = []
                for argument in node.args:
                    value = self.expression(argument.value if isinstance(argument, ast.Starred) else argument)
                    if isinstance(argument, ast.Starred):
                        if type(value.constant) not in (tuple, list) or len(value.constant) + len(values) > 128:
                            return self.unsupported(node, "Star arguments for layout cannot be safely bound", [receiver, value])
                        values.extend(value.constant)
                    else:
                        values.append(value.constant)
                if any(kw.arg is None for kw in node.keywords):
                    return self.unsupported(node, "Layout keyword unpacking is unproven", [receiver])
                options = {kw.arg: self.expression(kw.value).constant for kw in node.keywords}
                if len(values) == 1 and type(values[0]) in (tuple, list):
                    values = list(values[0])
                dimensions = [value.expression if isinstance(value, Symbol) else value for value in values]
                check = transform(self.tensor_spec(receiver), name, dimensions)
                if name in {"to", "type", "type_as", "float", "half"}:
                    try:
                        symbolic = SourceEvaluator().method(receiver.symbolic, name, values, options, node)
                        check.spec = TensorSpec(symbolic.spec()["shape"], symbolic.dtype, "cast_result")
                        check.status, check.reason, check.constraints = "proven", "", []
                    except (SourceBoundary, AttributeError, TypeError, ValueError):
                        check.spec = TensorSpec(None, None, "unresolved_cast")
                        check.status, check.reason = "unknown", "Cast dtype/device arguments are unproven"
                elif options:
                    check.spec = TensorSpec(None, None, "unresolved_layout")
                    check.status, check.reason = "unknown", "Layout keyword arguments are not yet bound"
                value = self.event(node, None, "reshape", name.title(), {"input": receiver},
                                   {"method": name, "dimensions": [v if type(v) in (int, str) else "unknown" if v is UNKNOWN else str(v) for v in dimensions],
                                    "keyword_arguments": bool(options)})
                return self.attach_shape(value, self.flow.events[-1], check)
            if any(self.global_symbol(node.func) is item for item in (enumerate, iter)) and node.args:
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
            descriptor = self.canonical_slice(node, arguments)
            if descriptor is not None:
                return descriptor
            from torch.nn import functional as functional
            target = self.global_symbol(node.func)
            if target is UNKNOWN:
                target = function.constant
            if target is math.sqrt and len(positional) == 1 and not keywords and type(positional[0].constant) in (int, float) and positional[0].constant >= 0:
                self.resolved(node, "scalar_math_sqrt", input=positional[0].constant)
                return StaticValue.literal(math.sqrt(positional[0].constant))
            if any(target is conversion for conversion in (float, int, bool)) and len(positional) == 1 and not keywords and type(positional[0].constant) in (int, float, bool):
                value = target(positional[0].constant)
                if type(value) is float and not math.isfinite(value):
                    return StaticValue()
                self.resolved(node, "native_scalar_conversion", target=target.__name__, value=value)
                return StaticValue.literal(value)
            if target is functional.dropout and positional:
                probability = keywords.get("p", positional[1] if len(positional) > 1 else StaticValue.literal(0.5)).constant
                training = keywords.get("training", positional[2] if len(positional) > 2 else StaticValue.literal(True)).constant
                if type(probability) in (int, float) and 0 <= probability <= 1 and type(training) is bool and keywords.keys() <= {"p", "training", "inplace"} and keywords.get("inplace", StaticValue.literal(False)).constant is False:
                    value = self.event(node, None, "dropout", "Dropout", {"input": positional[0]}, {"probability": probability, "training": training})
                    check = transform(self.tensor_spec(positional[0]), "contiguous", [])
                    check.rule = "native_functional_dropout"
                    return self.attach_shape(value, self.flow.events[-1], check)
            if any(target is primitive for primitive in (functional.gelu, functional.relu, functional.silu,
                                                         torch.tanh, torch.sigmoid, torch.rsqrt, torch.erf,
                                                         torch.pow, functional.softmax)) and positional:
                value = self.event(node, None, "activation", "Activation", arguments)
                self.flow.events[-1].metadata["native_callable"] = f"{target.__module__}.{target.__name__}"
                check = transform(self.tensor_spec(positional[0]), "contiguous", [])
                check.rule = "native_elementwise_activation"
                if target is functional.softmax:
                    axis = keywords.get("dim", positional[1] if len(positional) > 1 else StaticValue()).constant
                    self.flow.events[-1].metadata["normalization_axis"] = axis if type(axis) is int else None
                    if check.spec.shape is None or type(axis) is not int or not -len(check.spec.shape) <= axis < len(check.spec.shape):
                        check.spec.shape = None
                        check.status, check.reason = "unknown", "Softmax axis is unproven"
                    if "dtype" in keywords:
                        check.spec.dtype = DTYPES.get(keywords["dtype"].constant)
                    if keywords.keys() - {"dim", "dtype", "_stacklevel"} or len(positional) > 2:
                        check.status, check.reason = "unknown", "Softmax arguments are not fully bound"
                elif target is torch.pow:
                    if len(positional) == 2 and not keywords:
                        if positional[1].domain == "tensor":
                            check = broadcast([self.tensor_spec(item) for item in positional], "native_pow_broadcast")
                        elif type(positional[1].constant) not in (int, float):
                            check.status, check.reason = "unknown", "Exponent is not an established tensor or scalar"
                    else:
                        check.status, check.reason = "unknown", "Pow arguments are not fully bound"
                elif target is functional.gelu:
                    if (len(positional) != 1 or keywords.keys() - {"approximate"}
                            or type(keywords.get("approximate", StaticValue.literal("none")).constant) is not str
                            or keywords.get("approximate", StaticValue.literal("none")).constant not in {"none", "tanh"}):
                        check.status, check.reason = "unknown", "GELU arguments are not fully bound"
                elif target in (functional.relu, functional.silu):
                    if (len(positional) != 1 or keywords.keys() - {"inplace"}
                            or keywords.get("inplace", StaticValue.literal(False)).constant is not False):
                        check.status, check.reason = "unknown", "In-place activation effects or arguments are unproven"
                elif len(positional) != 1 or keywords:
                    check.status, check.reason = "unknown", "Unary activation arguments are not fully bound"
                if check.spec.dtype not in {"F16", "BF16", "F32", "F64"}:
                    check.status, check.reason = "unknown", "Activation dtype compatibility is unproven"
                if check.status in {"unknown", "invalid"}:
                    check.spec = TensorSpec(None, None, "unproven_activation")
                return self.attach_shape(value, self.flow.events[-1], check)
            if function.module_path is not None and function.method is None:
                self.flow.call_sites.append({"module_path": function.module_path,
                                             **self.source_location(node)})
                return self.event(node, function.module_path, "module", function.module_path, arguments,
                                  self.module_argument_roles(function.module_path, arguments))
            if function.method is not None and self.depth < 3:
                return self.helper(function.module_path, function.method, positional, node, keywords)
            from transformers.pytorch_utils import apply_chunking_to_forward
            if target is apply_chunking_to_forward and len(positional) >= 4 and positional[1].constant == 0 and not keywords:
                helper = arguments["0"]
                if helper.method:
                    self.resolved(node, "native_zero_chunk_passthrough", chunk_size=0,
                        source_sha256=hashlib.sha256(inspect.getsource(apply_chunking_to_forward).encode()).hexdigest())
                    return self.helper(helper.module_path, helper.method, positional[3:], node, keywords)
            if target is TORCH_PRIMITIVES["arange"]:
                result = self.event(node, None, "source_operation", "Position range", arguments)
                if not keywords.keys() - {"device", "dtype"}:
                    bounds = [arg.constant for arg in positional]
                    if len(bounds) == 2 and type(bounds[0]) is int and bounds[0] == 0:
                        bounds = bounds[1:]
                    if len(bounds) == 1 and (isinstance(bounds[0], Symbol) or type(bounds[0]) is int):
                        dtype = DTYPES.get(keywords["dtype"].constant) if "dtype" in keywords else "I64"
                        if dtype:
                            result.symbolic = Tensor((bounds[0],), dtype, f"arange({bounds[0].expression if isinstance(bounds[0], Symbol) else bounds[0]})")
                            result.domain = "tensor"
                            result.shape_spec = TensorSpec(result.symbolic.spec()["shape"], dtype, "position_range")
                            self.flow.events[-1].metadata["proven_output"] = result.symbolic.spec()
                            self.flow.events[-1].metadata["shape_check"] = {"rule": "native_arange", "status": "conditional",
                                "constraints": [f"{bounds[0]} >= 0"], "output_shape": result.shape_spec.shape,
                                "output_dtype": dtype, "runtime_verified": False}
                            self.resolved(node, "symbolic_arange", output=result.symbolic.spec())
                return result
            native_matrix = next((method for method in ("matmul", "bmm", "einsum", "cat", "stack") if target is getattr(torch, method)), None)
            if native_matrix is not None:
                name = native_matrix
                kind = "matmul" if name in {"matmul", "bmm", "einsum"} else "concatenation"
                dimension = arguments.get("dim", arguments.get("1", StaticValue(constant=0))).constant
                metadata = {"method": name}
                if name in {"cat", "stack"} and type(dimension) is int:
                    metadata["dimension"] = dimension
                value = self.event(
                    node,
                    None,
                    kind,
                    "Matrix multiply" if kind == "matmul" else "Concatenate",
                    arguments,
                    metadata,
                )
                if name in {"matmul", "bmm"} and len(positional) == 2:
                    return self.attach_shape(value, self.flow.events[-1], matmul(self.tensor_spec(positional[0]), self.tensor_spec(positional[1]), bmm=name == "bmm"))
                if name in {"cat", "stack"} and positional and positional[0].items is not None:
                    self.flow.events[-1].arguments = {str(index): item for index, item in enumerate(positional[0].items)}
                    check = concatenate([self.tensor_spec(item) for item in positional[0].items], dimension, stack=name == "stack")
                    return self.attach_shape(value, self.flow.events[-1], check)
                return value
            if any(target is item for item in (hasattr, isinstance)):
                if (self.global_symbol(node.func) is isinstance and len(positional) == 2
                        and positional[0].domain == "device_type" and positional[1].constant is str):
                    self.resolved(node, "native_device_type_is_string", runtime_value_known=False)
                    return StaticValue.literal(True)
                if target is hasattr and len(node.args) == 2:
                    obj, attribute = arguments["0"], arguments["1"].constant
                    if obj.module_path is not None and isinstance(attribute, str):
                        module = self.model.get_submodule(obj.module_path) if obj.module_path else self.model
                        if any(attribute in registry for registry in (vars(module), module._modules, module._parameters, module._buffers)):
                            return StaticValue.literal(True)
                        known = inspect.getattr_static(module, attribute, UNKNOWN)
                        return StaticValue() if isinstance(known, property) else StaticValue.literal(known is not UNKNOWN)
                return StaticValue()
            external = self.external_helper(node, positional, keywords, arguments)
            if external is not None:
                return external
            from transformers.utils import ModelOutput
            output_type = self.global_symbol(node.func)
            if isinstance(output_type, type) and issubclass(output_type, ModelOutput) and dataclasses.is_dataclass(output_type):
                names = [field.name for field in dataclasses.fields(output_type)]
                if len(positional) <= len(names) and not (set(names[:len(positional)]) & keywords.keys()) and not keywords.keys() - set(names):
                    fields = {name: value for name, value in zip(names, positional)}
                    fields.update(keywords)
                    ordered = {name: fields.get(name, StaticValue.literal(None)) for name in names}
                    value = combine(list(ordered.values()))
                    value.fields, value.domain = ordered, "model_output"
                    value.items = [item for item in ordered.values() if item.constant is not None]
                    self.resolved(node, "native_model_output_fields", output_type=f"{output_type.__module__}.{output_type.__qualname__}", fields=names)
                    return value
            if isinstance(node.func, ast.Attribute):
                result = combine([receiver, *arguments.values()])
                if result.refs:
                    self.flow.warnings.append(
                        f"Opaque call {ast.unparse(node.func)}: dependencies retained, transformation unverified"
                    )
                    return self.unsupported(node, "Opaque call transformation is unverified", [receiver, *arguments.values()])
                return self.deferred_unknown_call(node)
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
            return self.deferred_unknown_call(node)
        return self.unsupported(node, "Expression semantics are not supported by the static interpreter")

    def external_helper(self, node: ast.Call, positional: list[StaticValue],
                        keywords: dict[str, StaticValue], arguments: dict[str, StaticValue]) -> StaticValue | None:
        """Summarize an external Python helper only after all symbolic source steps are proven."""
        function = self.global_symbol(node.func)
        if function is UNKNOWN and isinstance(node.func, ast.Name):
            function = self.env.get(node.func.id, StaticValue()).constant
        if not inspect.isfunction(function) and not any(function is primitive for primitive in TORCH_PRIMITIVES.values()):
            return None
        if inspect.isfunction(function) and any(value.refs or value.module_path is not None for value in positional):
            inlined = self.inline_function(function, positional, keywords, node)
            if inlined is not None:
                return inlined
        return self.source_summary(node, function, positional, keywords, arguments)

    def inline_function(self, function: Any, positional: list[StaticValue], keywords: dict[str, StaticValue], node: ast.Call) -> StaticValue | None:
        """Interpret a selected plain Python helper in the same dependency graph."""
        if self.depth >= 6 or function.__closure__ or hasattr(function, "__wrapped__"):
            return None
        try:
            text = textwrap.dedent(inspect.getsource(function))
            tree = next(item for item in ast.parse(text).body if isinstance(item, ast.FunctionDef))
            if tree.decorator_list or len(text) > 131072 or sum(1 for _ in ast.walk(tree)) > 4096 or any(isinstance(item, ast.Raise) for item in ast.walk(tree)):
                return None
            bound = self.bind_helper(tree, positional, keywords, skip_self=False)
            if bound is None:
                return None
        except (OSError, TypeError, SyntaxError, StopIteration):
            return None
        old = self.env, self.returned, self.source_globals, self.source_name, self.source_hash, self.flow.outputs
        self.env, self.returned, self.source_globals = bound, [], function.__globals__
        self.source_name = f"{function.__module__}.{function.__qualname__}"
        self.source_hash = hashlib.sha256(text.encode()).hexdigest()
        provenance = self.source_location(tree)
        self.collect_source_calls(tree)
        self.depth += 1
        try:
            self.statements(tree.body)
            result = merge_outputs(self.returned)
        finally:
            self.depth -= 1
            self.env, self.returned, self.source_globals, self.source_name, self.source_hash, self.flow.outputs = old
        self.resolved(node, "selected_python_helper", helper=provenance,
                      argument_names=sorted(bound), tensor_execution=False)
        return result

    def source_summary(self, node: ast.Call, function: Any, positional: list[StaticValue],
                       keywords: dict[str, StaticValue], arguments: dict[str, StaticValue],
                       owner: ModuleState | None = None) -> StaticValue | None:
        """Share symbolic primitive and helper semantics between free and bound calls."""
        evaluator = SourceEvaluator()

        def convert(value: StaticValue) -> Any:
            if value.symbolic is not None:
                return value.symbolic
            if value.constant is not UNKNOWN:
                return value.constant
            if value.items is not None:
                return tuple(convert(item) for item in value.items)
            if value.fields is not None:
                return {key: convert(item) for key, item in value.fields.items()}
            raise SourceBoundary("Helper argument has no established symbolic value")

        try:
            result = evaluator.invoke(function, ([owner] if owner is not None else []) + [convert(value) for value in positional],
                                      {name: convert(value) for name, value in keywords.items()}, node)
            proof = evaluator.evidence()
            proof["inputs"] = {
                name: value.symbolic.spec() if value.symbolic is not None else {"value": value.constant}
                for name, value in arguments.items()
                if value.symbolic is not None or type(value.constant) in (type(None), bool, int, float, str)
            }
            proof["input_basis"] = "Symbolic input convention, native embedding signature and recorded source layout operations; not runtime observations"
            if isinstance(result, Tensor):
                output = result.spec()
                value = self.event(node, None, "source_operation", ast.unparse(node.func).replace("_", " ").capitalize(),
                                   arguments, {"source_proof": proof, "proven_output": output,
                                               "argument_roles": {name: name for name in arguments}})
                value.symbolic = result
                value.domain = "tensor"
                value.shape_spec = TensorSpec(output["shape"], output["dtype"], "source_result")
            elif type(result) in (type(None), bool, int, float, str):
                if (type(result) is float and not math.isfinite(result)
                        or type(result) is int and result.bit_length() > 64
                        or type(result) is str and len(result) > 4096):
                    raise SourceBoundary("Helper scalar output exceeds the literal budget")
                value = StaticValue.literal(result)
            elif isinstance(result, (FloatInfo, Symbol)):
                value = StaticValue(constant=result, domain="metadata")
            elif type(result) in (tuple, list):
                def metadata_only(item: Any) -> bool:
                    return (type(item) in (type(None), bool, int, float, str) or isinstance(item, Symbol)
                            or type(item) in (tuple, list) and all(metadata_only(part) for part in item))
                if not metadata_only(result):
                    raise SourceBoundary("Tensor container outputs require explicit source ports")
                value = StaticValue.literal(result)
            else:
                raise SourceBoundary("Helper output is not a supported tensor or scalar")
            self.resolved(node, "external_python_helper", proof=proof,
                          **({"output": output} if isinstance(result, Tensor)
                             else {"value": result} if type(result) in (type(None), bool, int, float, str)
                             else {"result_type": type(result).__name__}))
            return value
        except (SourceBoundary, AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError, ZeroDivisionError, RecursionError) as exc:
            self.helper_boundaries[(self.source_name, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))] = {
                "reason": str(exc), "partial_proof": evaluator.evidence(), "established": False,
            }
            self.resolved(node, "symbolic_bound_helper_attempt" if owner is not None else "external_python_helper_boundary", established=False, reason=str(exc),
                          partial_proof=evaluator.evidence())
        return None

    def module_state(self, module: torch.nn.Module) -> ModuleState:
        """Expose bounded native attributes and the verified native dtype property."""
        from transformers.modeling_utils import ModuleUtilsMixin

        attributes = {name: value for name, value in vars(module).items()
                      if type(value) in (bool, int, float, str, type(None))}
        if "config" in vars(module):
            attributes["config"] = vars(module)["config"]
        evidence = {"module_type": f"{type(module).__module__}.{type(module).__qualname__}"}
        if inspect.getattr_static(type(module), "dtype", None) is ModuleUtilsMixin.dtype:
            parameter = next(module.named_parameters(), None)
            if parameter is not None:
                attributes["dtype"] = parameter[1].dtype
                evidence.update(dtype_parameter=parameter[0], dtype=str(parameter[1].dtype),
                                dtype_property_sha256=hashlib.sha256(inspect.getsource(ModuleUtilsMixin.dtype.fget).encode()).hexdigest())
        return ModuleState(attributes, evidence)

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
        arguments = {str(index): value for index, value in enumerate(values)}
        arguments.update(keywords or {})
        summary = self.source_summary(node, method, values, keywords or {}, arguments, self.module_state(module))
        if summary is not None:
            return summary
        from .diagnostic_effects import diagnostic_effects

        diagnostic_arguments = {}
        for name, value in bound.items():
            if value.symbolic is not None:
                diagnostic_arguments[name] = value.symbolic
            elif value.constant is not UNKNOWN:
                diagnostic_arguments[name] = value.constant
            else:
                break
        else:
            proof = diagnostic_effects(method, self.module_state(module), diagnostic_arguments)
            if proof is not None:
                detail = {"kind": "conditional_logging", "message": "Native source may emit a data-dependent diagnostic; content was not evaluated", "proof": proof}
                self.flow.diagnostics.append(detail)
                self.flow.profile_assumptions.extend(proof["profile_assumptions"])
                self.resolved(node, "diagnostic_only_helper", proof=proof)
                return StaticValue.literal(None)
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
                    keywords: dict[str, StaticValue], skip_self: bool = True) -> dict[str, StaticValue] | None:
        """Bind AST parameter names, defaults and variadics without executing a helper."""
        args = function.args
        positional = args.posonlyargs + args.args
        if skip_self and positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        if len(values) > len(positional) and args.vararg is None:
            return None
        bound = {arg.arg: value for arg, value in zip(positional, values)}
        allowed = {arg.arg for arg in args.args + args.kwonlyargs} - ({"self", "cls"} if skip_self else set())
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
        defaults = argument_defaults(function)
        for arg in positional + args.kwonlyargs:
            if arg.arg in bound:
                continue
            if arg.arg not in defaults:
                return None
            value = literal_constant(defaults[arg.arg])
            bound[arg.arg] = StaticValue(may_be_none=True) if value is UNKNOWN else StaticValue.literal(value)
        if args.vararg:
            bound[args.vararg.arg] = StaticValue(items=values[len(positional):])
        if args.kwarg:
            bound[args.kwarg.arg] = StaticValue(fields=extras)
        return bound
