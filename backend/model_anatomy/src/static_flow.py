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
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils.import_utils import is_torch_fx_proxy, is_torchdynamo_compiling

from .inputs import InputResolver
from .ir import TensorSpec
from .tensor_contracts import tensor_outputs

UNKNOWN = object()
DTYPES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.bool: "BOOL",
}
MAX_STATIC_ITERATIONS = 128
_TORCH_FINFO = torch.finfo
_TORCH_FULL = torch.full
_TORCH_TRIU = torch.triu
_FINFO_MIN = {
    torch.float16: -65504.0,
    torch.bfloat16: -3.3895313892515355e38,
    torch.float32: -3.4028234663852886e38,
    torch.float64: -1.7976931348623157e308,
}


@dataclass
class StaticValue:
    refs: list[tuple[str, str]] = field(default_factory=list)
    constant: Any = UNKNOWN
    module_path: str | None = None
    method: str | None = None
    items: list[StaticValue] | None = None
    fields: dict[str, StaticValue] | None = None
    spec: TensorSpec | None = None
    symbol: str | None = None
    port_specs: dict[str, TensorSpec] | None = None
    category: str | None = None


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
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


def combine(values: list[StaticValue]) -> StaticValue:
    return StaticValue(list(dict.fromkeys(ref for value in values for ref in value.refs)))


def join_values(values: list[StaticValue]) -> StaticValue:
    result = combine(values)
    if values and all(value.spec is not None for value in values):
        first = values[0].spec
        if all(value.spec.shape == first.shape and value.spec.dtype == first.dtype for value in values):
            result.spec = TensorSpec(first.shape, first.dtype, "proven_branch_join")
    if values and values[0].symbol is not None and all(value.symbol == values[0].symbol for value in values):
        result.symbol = values[0].symbol
    if (
        values
        and type(values[0].constant) in (type(None), bool, int, float, str)
        and all(
            type(value.constant) is type(values[0].constant) and value.constant == values[0].constant
            for value in values
        )
    ):
        result.constant = values[0].constant
    return result


def merge_outputs(values: list[StaticValue]) -> StaticValue:
    if len(values) == 1:
        return values[0]
    result = join_values(values)
    field_names = set.intersection(*(set(value.fields or {}) for value in values)) if values else set()
    if field_names:
        result.fields = {name: join_values([value.fields[name] for value in values]) for name in field_names}
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
        self.inputs = InputResolver(model)
        self.iterations = 0
        self.positive_symbols: set[str] = set()
        self.source_method_name: str | None = None
        self.source_method: Any = None

    def analyze(self) -> StaticFlow:
        try:
            method = inspect.unwrap(type(self.module).forward)
            self.source_globals = method.__globals__
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            function = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        except (OSError, TypeError, SyntaxError, StopIteration) as exc:
            self.flow.warnings.append(f"Forward source unavailable: {type(exc).__name__}")
            return self.flow
        self.source_method_name, self.source_method = "forward", method
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
        spec = self.inputs.spec(name, self.path)
        value = StaticValue([(f"input:{name}", "output")], spec=spec)
        self.flow.inputs[name] = value
        self.positive_symbols.update(size for size in spec.shape or [] if type(size) is str)
        assumption = {
            "code": "canonical_input_profile",
            "input": name,
            "shape": spec.shape,
            "dtype": spec.dtype,
            "positive_symbolic_axes": [size for size in spec.shape or [] if type(size) is str],
            "basis": "Canonical nonempty local graph input profile; native meta dtype, no autocast; not runtime evidence",
        }
        if assumption not in self.flow.diagnostics:
            self.flow.diagnostics.append(assumption)
        return value

    def event(
        self,
        node: ast.AST,
        path: str | None,
        kind: str,
        label: str,
        arguments: dict[str, StaticValue],
        metadata: dict[str, Any] | None = None,
        *,
        native_container_member: bool = False,
    ) -> StaticValue:
        metadata = dict(metadata or {})
        if path is not None and "output_port_specs" not in metadata:
            spec = self.module_spec(path, arguments)
            if spec is not None:
                metadata["output_port_specs"] = {"output": self.spec_data(spec)}
                metadata["contract"] = "exact_native_torch_primitive.v1"
            elif native_container_member:
                from .native_output_contracts import CONTRACT_SOURCE_SHA256, native_module_outputs

                outputs = self.container_member_outputs(path, arguments, native_module_outputs)
                if outputs is not None:
                    container = self.model.get_submodule(path) if path else self.model
                    metadata["output_port_specs"] = {
                        port: self.spec_data(value) for port, value in outputs.items()
                    }
                    metadata["contract"] = "reviewed_native_module_list_member.v1"
                    metadata["native_container"] = path
                    metadata["native_member_classes"] = sorted(
                        {f"{type(child).__module__}.{type(child).__name__}" for child in container}
                    )
                    metadata["native_contract_source_hashes"] = CONTRACT_SOURCE_SHA256
            else:
                from .native_output_contracts import CONTRACT_SOURCE_SHA256, native_module_outputs

                module = self.model.get_submodule(path) if path else self.model
                incoming = self.module_input(path, arguments).spec
                outputs = native_module_outputs(module, incoming) if incoming is not None else None
                if outputs is not None:
                    metadata["output_port_specs"] = {
                        port: self.spec_data(value) for port, value in outputs.items()
                    }
                    metadata["contract"] = "reviewed_native_output.v1"
                    metadata["native_class"] = f"{type(module).__module__}.{type(module).__name__}"
                    metadata["native_contract_source_hashes"] = CONTRACT_SOURCE_SHA256
                elif type(module) in {
                    torch.nn.Linear,
                    torch.nn.Embedding,
                    torch.nn.Dropout,
                    torch.nn.Identity,
                    torch.nn.GELU,
                    torch.nn.ReLU,
                    torch.nn.SiLU,
                    torch.nn.Tanh,
                    torch.nn.Sigmoid,
                    torch.nn.LayerNorm,
                }:
                    metadata["output_contract_required"] = True
        self.counter += 1
        identifier = f"event{self.counter}"
        self.flow.events.append(
            FlowEvent(identifier, path, kind, label, arguments, getattr(node, "lineno", 0), metadata)
        )
        ports = (metadata or {}).get("output_port_specs")
        if ports:
            outputs = {port: TensorSpec(**spec) for port, spec in ports.items()}
            if "output" in outputs:
                return StaticValue([(identifier, "output")], spec=outputs["output"])
            if (metadata or {}).get("output_arity") == len(outputs):
                items = [
                    StaticValue([(identifier, str(index))], spec=outputs[str(index)])
                    for index in range(len(outputs))
                ]
                return StaticValue(
                    [ref for item in items for ref in item.refs],
                    items=items,
                    port_specs=outputs,
                    category="tuple",
                )
            return StaticValue([(identifier, "output")], port_specs=outputs)
        if path is not None:
            return StaticValue([(identifier, "output")], spec=self.module_spec(path, arguments))
        return StaticValue([(identifier, "output")])

    def container_member_outputs(
        self,
        path: str,
        arguments: dict[str, StaticValue],
        native_module_outputs: Any,
    ) -> dict[str, TensorSpec] | None:
        """Prove a common output map for every member of a native ModuleList loop."""
        container = self.model.get_submodule(path) if path else self.model
        if (
            type(container) is not torch.nn.ModuleList
            or not container._modules
            or len(container._modules) > MAX_STATIC_ITERATIONS
            or container.training
            or "forward" in vars(container)
            or container._forward_hooks
            or container._forward_pre_hooks
            or container._backward_hooks
        ):
            return None
        common: dict[str, TensorSpec] | None = None
        for name, child in container._modules.items():
            if not isinstance(child, torch.nn.Module):
                return None
            child_path = f"{path}.{name}" if path else name
            incoming = self.module_input(child_path, arguments).spec
            if incoming is None:
                return None
            outputs = native_module_outputs(child, incoming)
            if outputs is None or common is not None and outputs != common:
                return None
            common = outputs
        return common

    @staticmethod
    def spec_data(spec: TensorSpec) -> dict[str, Any]:
        return {"shape": spec.shape, "dtype": spec.dtype, "semantic_name": spec.semantic_name}

    def module_input(self, path: str, arguments: dict[str, StaticValue]) -> StaticValue:
        module = self.model.get_submodule(path) if path else self.model
        try:
            signature = inspect.signature(module.forward)
            positional = [arguments[str(index)] for index in range(sum(key.isdecimal() for key in arguments))]
            bound = signature.bind(
                *positional, **{key: value for key, value in arguments.items() if not key.isdecimal()}
            )
        except (TypeError, ValueError, KeyError):
            return StaticValue()
        first = next(iter(signature.parameters), None)
        value = bound.arguments.get(first)
        return value if isinstance(value, StaticValue) else StaticValue()

    def module_spec(self, path: str, arguments: dict[str, StaticValue]) -> TensorSpec | None:
        """Transfer facts through exact Torch primitives only; custom modules are not shape identities."""
        module = self.model.get_submodule(path) if path else self.model
        if (
            "forward" in vars(module)
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
        ):
            return None
        value = self.module_input(path, arguments).spec
        if value is None:
            return None
        if type(module) is torch.nn.Linear:
            dtype = DTYPES.get(module.weight.dtype)
            shape = list(value.shape) if value.shape is not None else None
            if (
                not shape
                or shape[-1] != module.in_features
                or dtype != value.dtype
                or tuple(module.weight.shape) != (module.out_features, module.in_features)
                or module.bias is not None
                and (
                    module.bias.dtype != module.weight.dtype
                    or tuple(module.bias.shape) != (module.out_features,)
                )
            ):
                return None
            shape[-1] = module.out_features
            return TensorSpec(shape, dtype, "native_linear_output")
        if type(module) is torch.nn.Embedding:
            if value.dtype not in {"I32", "I64"}:
                return None
            shape = [*value.shape, module.embedding_dim] if value.shape is not None else None
            return TensorSpec(shape, DTYPES.get(module.weight.dtype), "native_embedding_output")
        if type(module) in {
            torch.nn.Dropout,
            torch.nn.Identity,
            torch.nn.GELU,
            torch.nn.ReLU,
            torch.nn.SiLU,
            torch.nn.Tanh,
            torch.nn.Sigmoid,
        }:
            if type(module) not in {torch.nn.Dropout, torch.nn.Identity} and value.dtype not in {
                "F16",
                "BF16",
                "F32",
                "F64",
            }:
                return None
            return TensorSpec(value.shape, value.dtype, "native_unary_output")
        if (
            type(module) is torch.nn.LayerNorm
            and value.dtype in {"F16", "BF16", "F32", "F64"}
            and value.shape is not None
            and list(value.shape[-len(module.normalized_shape) :]) == list(module.normalized_shape)
            and all(
                parameter is None
                or (
                    DTYPES.get(parameter.dtype) == value.dtype
                    and tuple(parameter.shape) == tuple(module.normalized_shape)
                )
                for parameter in (module.weight, module.bias)
            )
            and type(module.eps) in (int, float)
            and math.isfinite(module.eps)
            and module.eps > 0
        ):
            return TensorSpec(value.shape, value.dtype, "native_layer_norm_output")
        return None

    @staticmethod
    def dimension_value(value: int | str) -> StaticValue:
        return StaticValue(constant=value) if type(value) is int else StaticValue(symbol=value)

    @staticmethod
    def primitive(value: Any) -> StaticValue:
        if type(value) in (type(None), bool, int, str) or type(value) is float and math.isfinite(value):
            return StaticValue(constant=value)
        if type(value) in (tuple, list) and len(value) <= MAX_STATIC_ITERATIONS:
            items = [StaticFlowAnalyzer.primitive(item) for item in value]
            return StaticValue(items=items, category="tuple" if type(value) is tuple else "list")
        return StaticValue()

    def iterable_items(self, value: StaticValue) -> list[StaticValue] | None:
        if value.items is not None:
            return value.items
        if type(value.constant) in (tuple, list):
            return self.primitive(value.constant).items
        if value.module_path is not None:
            module = self.model.get_submodule(value.module_path) if value.module_path else self.model
            if (
                type(module) in {torch.nn.ModuleList, torch.nn.Sequential}
                and len(module._modules) <= MAX_STATIC_ITERATIONS
            ):
                return [
                    StaticValue(module_path=f"{value.module_path}.{name}".lstrip("."))
                    for name in module._modules
                ]
        return None

    def builtin_metadata(self, node: ast.Call, arguments: dict[str, StaticValue]) -> StaticValue | None:
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            return None
        function = self.source_callable(node.func)
        if function is getattr and len(node.args) in {2, 3} and not node.keywords:
            obj, name = arguments["0"].constant, arguments["1"].constant
            from transformers import PretrainedConfig

            if isinstance(obj, PretrainedConfig) and type(name) is str:
                values = vars(obj)
                if name in values:
                    result = self.primitive(values[name])
                elif inspect.getattr_static(type(obj), name, UNKNOWN) is not UNKNOWN:
                    from .native_metadata_contracts import native_config_attribute

                    supported, value = native_config_attribute(obj, name)
                    if not supported:
                        return None
                    result = self.primitive(value)
                elif len(node.args) == 3:
                    result = arguments["2"]
                else:
                    return None
                self.flow.diagnostics.append(
                    {
                        "code": "native_config_attribute",
                        "attribute": name,
                        "basis": "Inspected native config instance; no descriptor invocation",
                    }
                )
                return result
        if function is len and len(node.args) == 1 and not node.keywords:
            value = arguments["0"]
            items = self.iterable_items(value)
            if items is not None:
                return StaticValue(constant=len(items))
            if value.spec is not None and value.spec.shape:
                return self.dimension_value(value.spec.shape[0])
            if type(value.constant) in (str, dict):
                return StaticValue(constant=len(value.constant))
        if function is range and 1 <= len(node.args) <= 3 and not node.keywords:
            values = [arguments[str(index)].constant for index in range(len(node.args))]
            if all(type(value) is int and abs(value) <= 1_000_000 for value in values):
                try:
                    sequence = range(*values)
                    if len(sequence) <= MAX_STATIC_ITERATIONS:
                        return StaticValue(items=[StaticValue(constant=value) for value in sequence])
                except ValueError:
                    pass
        if function is zip and not node.keywords and node.args:
            sequences = [self.iterable_items(arguments[str(index)]) for index in range(len(node.args))]
            if all(sequence is not None for sequence in sequences):
                items = [StaticValue(items=list(row)) for row in zip(*sequences)]
                if len(items) <= MAX_STATIC_ITERATIONS:
                    return StaticValue(items=items)
        if function in (enumerate, iter) and node.args and not node.keywords:
            value = arguments["0"]
            if len(node.args) == 1 and value.module_path is not None:
                return value
            items = self.iterable_items(value)
            if function is iter and len(node.args) == 1 and items is not None:
                return StaticValue(items=items)
            start = arguments.get("1", StaticValue(constant=0)).constant
            if function is enumerate and len(node.args) <= 2 and items is not None and type(start) is int:
                return StaticValue(
                    items=[
                        StaticValue(items=[StaticValue(constant=index), item])
                        for index, item in enumerate(items, start)
                    ]
                )
        return None

    def tensor_event(
        self,
        node: ast.Call,
        method: str,
        values: dict[str, StaticValue],
        parameters: dict[str, Any],
        callable_name: str,
    ) -> StaticValue | None:
        if any(value.spec is None for value in values.values()):
            return None
        outputs = tensor_outputs(method, {name: value.spec for name, value in values.items()}, parameters)
        if outputs is None:
            return None
        metadata = {
            "method": method,
            "parameters": parameters,
            "callable": callable_name,
            "contract": f"torch_tensor.{method}.v1",
            "torch_version": str(torch.__version__),
            "runtime_verified": False,
            "output_port_specs": {port: self.spec_data(spec) for port, spec in outputs.items()},
        }
        if method == "chunk":
            metadata["output_arity"] = len(outputs)
        return self.event(
            node, None, "split" if method == "chunk" else "tensor_operation", method.title(), values, metadata
        )

    def tensor_method(
        self, node: ast.Call, receiver: StaticValue, arguments: dict[str, StaticValue]
    ) -> StaticValue | None:
        if receiver.spec is None or not isinstance(node.func, ast.Attribute):
            return None
        name = node.func.attr
        has_starred = any(isinstance(arg, ast.Starred) for arg in node.args)
        has_expansion = any(keyword.arg is None for keyword in node.keywords)
        if (has_starred or has_expansion) and name not in {"view", "reshape"}:
            return None
        if has_expansion:
            return None
        shape = receiver.spec.shape
        if name in {"dim", "ndimension"} and not arguments and shape is not None:
            return StaticValue(constant=len(shape))
        if name == "size" and shape is not None and not node.keywords and len(node.args) <= 1:
            if not node.args:
                return StaticValue(items=[self.dimension_value(value) for value in shape], category="tuple")
            axis = arguments["0"].constant
            if type(axis) is int and -len(shape) <= axis < len(shape):
                return self.dimension_value(shape[axis])
        if name == "numel" and not arguments and shape is not None and all(type(dim) is int for dim in shape):
            return StaticValue(constant=math.prod(shape))
        contracts = {
            "chunk": [("chunks", UNKNOWN), ("dim", 0)],
            "any": [("dim", None), ("keepdim", False)],
            "transpose": [("dim0", UNKNOWN), ("dim1", UNKNOWN)],
            "unsqueeze": [("dim", UNKNOWN)],
            "squeeze": [("dim", UNKNOWN)],
            "flatten": [("start_dim", 0), ("end_dim", -1)],
        }
        if name in contracts:
            parameters = self.bind_values(node, arguments, contracts[name])
            if parameters is not None:
                if name == "transpose":
                    parameters = {"dimensions": [parameters["dim0"], parameters["dim1"]]}
                return self.tensor_event(node, name, {"input": receiver}, parameters, f"torch.Tensor.{name}")
        if self.trusted_t5_tensor_source():
            from .native_metadata_contracts import trusted_torch_tensor_method
            if (
                name == "clone"
                and not node.args
                and not node.keywords
                and trusted_torch_tensor_method("clone")
            ):
                return self.tensor_event(node, "clone", {"input": receiver}, {}, "torch.Tensor.clone")
            if name == "expand" and not node.keywords and not has_starred:
                dimensions = [
                    arguments[str(index)].constant
                    if type(arguments[str(index)].constant) is int
                    else arguments[str(index)].symbol
                    for index in range(len(node.args))
                ]
                if len(dimensions) == 1 and arguments["0"].items is not None:
                    dimensions = [
                        value.constant if type(value.constant) is int else value.symbol
                        for value in arguments["0"].items
                    ]
                if (
                    trusted_torch_tensor_method("expand")
                    and dimensions
                    and all(type(value) is int or type(value) is str for value in dimensions)
                ):
                    return self.tensor_event(
                        node,
                        "expand",
                        {"input": receiver},
                        {"dimensions": dimensions, "positive_symbols": sorted(self.positive_symbols)},
                        "torch.Tensor.expand",
                    )
            if name == "masked_fill" and len(node.args) == 2 and not node.keywords:
                mask, fill_value = arguments["0"], arguments["1"].constant
                if trusted_torch_tensor_method("masked_fill") and mask.spec is not None:
                    return self.tensor_event(
                        node,
                        "masked_fill",
                        {"input": receiver, "mask": mask},
                        {"value": fill_value},
                        "torch.Tensor.masked_fill",
                    )
        if name in {"view", "reshape"} and not node.keywords:
            values = []
            for index, arg in enumerate(node.args):
                value = arguments[str(index)]
                if isinstance(arg, ast.Starred):
                    if value.items is None:
                        return None
                    values.extend(value.items)
                else:
                    values.append(value)
            if len(values) == 1 and values[0].items is not None:
                values = values[0].items
            dimensions = [value.constant if type(value.constant) is int else value.symbol for value in values]
            if dimensions and all(type(value) is int or type(value) is str for value in dimensions):
                return self.tensor_event(
                    node,
                    name,
                    {"input": receiver},
                    {"dimensions": dimensions, "positive_symbols": sorted(self.positive_symbols)},
                    f"torch.Tensor.{name}",
                )
        if name in {"float", "half", "long", "int", "bool"} and not arguments:
            dtype = {"float": "F32", "half": "F16", "long": "I64", "int": "I32", "bool": "BOOL"}[name]
            return self.tensor_event(
                node, "cast", {"input": receiver}, {"dtype": dtype}, f"torch.Tensor.{name}"
            )
        if name == "to" and (
            not node.args
            and set(arguments) in ({"dtype"}, {"device"})
            or len(node.args) == 1
            and not node.keywords
        ):
            from .native_metadata_contracts import trusted_torch_tensor_method

            if not trusted_torch_tensor_method("to"):
                return None
            target = arguments.get("dtype", arguments.get("device", arguments.get("0", StaticValue())))
            dtype = target.constant
            if isinstance(dtype, torch.dtype) and dtype in DTYPES:
                return self.tensor_event(
                    node, "cast", {"input": receiver}, {"dtype": DTYPES[dtype]}, "torch.Tensor.to"
                )
            if target.category == "torch_device" and receiver.spec.dtype is not None:
                return self.tensor_event(
                    node,
                    "cast",
                    {"input": receiver},
                    {"dtype": receiver.spec.dtype},
                    "torch.Tensor.to_device",
                )
        if name == "type_as" and len(node.args) == 1 and not node.keywords:
            other = arguments["0"].spec
            if other is not None and other.dtype is not None:
                return self.tensor_event(
                    node, "cast", {"input": receiver}, {"dtype": other.dtype}, "torch.Tensor.type_as"
                )
        if name in {"contiguous", "detach"} and not arguments:
            return StaticValue(list(receiver.refs), spec=receiver.spec)
        return None

    def trusted_t5_tensor_source(self) -> bool:
        if self.source_method_name == "invert_attention_mask":
            from .native_metadata_contracts import trusted_t5_module_utils_helper

            return trusted_t5_module_utils_helper(self.module, self.source_method_name, self.source_method)
        if self.source_method_name not in {"forward", "_prepare_4d_causal_attention_mask_with_cache_position"}:
            return False
        from .native_metadata_contracts import trusted_t5_stack_method

        return trusted_t5_stack_method(self.module, self.source_method_name, self.source_method)

    @staticmethod
    def bind_arguments(
        node: ast.Call, arguments: dict[str, StaticValue], fields: list[tuple[str, Any]]
    ) -> dict | None:
        parameters = [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if default is UNKNOWN else StaticValue(constant=default),
            )
            for name, default in fields
        ]
        try:
            bound = inspect.Signature(parameters).bind(
                *[arguments[str(index)] for index in range(len(node.args))],
                **{key: value for key, value in arguments.items() if not key.isdecimal()},
            )
            bound.apply_defaults()
        except (TypeError, ValueError):
            return None
        return dict(bound.arguments)

    @staticmethod
    def bind_values(
        node: ast.Call, arguments: dict[str, StaticValue], fields: list[tuple[str, Any]]
    ) -> dict | None:
        bound = StaticFlowAnalyzer.bind_arguments(node, arguments, fields)
        return {key: value.constant for key, value in bound.items()} if bound is not None else None

    def torch_operation(self, node: ast.Call, arguments: dict[str, StaticValue]) -> StaticValue | None:
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            return None
        function = self.source_callable(node.func)
        if self.trusted_t5_tensor_source() and function is _TORCH_FULL and torch.full is _TORCH_FULL:
            bound = self.bind_arguments(
                node, arguments,
                [("size", UNKNOWN), ("fill_value", UNKNOWN), ("dtype", UNKNOWN), ("device", UNKNOWN)],
            )
            if bound is not None:
                size = bound["size"].items
                dtype = bound["dtype"].constant
                fill = bound["fill_value"].constant
                dimensions = [item.constant if type(item.constant) is int else item.symbol for item in size or []]
                if (
                    size is not None
                    and len(dimensions) >= 2
                    and all(
                        type(axis) is int and 0 <= axis <= 1_000_000
                        or type(axis) is str and axis in self.positive_symbols
                        for axis in dimensions
                    )
                    and type(dtype) is torch.dtype
                    and dtype in _FINFO_MIN
                    and type(fill) in {int, float}
                    and (type(fill) is int or math.isfinite(fill))
                    and abs(fill) <= abs(_FINFO_MIN[dtype])
                    and bound["device"].category == "torch_device"
                ):
                    return self.event(
                        node, None, "tensor_creation", "Full", arguments,
                        {
                            "method": "full", "callable": "torch.full", "runtime_verified": False,
                            "contract": "torch_native_t5_full.v1",
                            "output_port_specs": {
                                "output": self.spec_data(TensorSpec(dimensions, DTYPES[dtype], "full"))
                            },
                        },
                    )
        if self.trusted_t5_tensor_source() and function is _TORCH_TRIU and torch.triu is _TORCH_TRIU:
            bound = self.bind_arguments(node, arguments, [("input", UNKNOWN), ("diagonal", 0)])
            if bound is not None:
                return self.tensor_event(
                    node, "triu", {"input": bound["input"]},
                    {"diagonal": bound["diagonal"].constant}, "torch.triu",
                )
        if (
            function is _TORCH_FINFO
            and torch.finfo is _TORCH_FINFO
            and len(node.args) == 1
            and not node.keywords
        ):
            dtype = arguments["0"].constant
            if type(dtype) is torch.dtype and dtype in _FINFO_MIN:
                return StaticValue(fields={"min": StaticValue(constant=_FINFO_MIN[dtype])})
        if function is torch.arange:
            if len(node.args) not in (1, 2) or not {key for key in arguments if not key.isdecimal()} <= {
                "dtype",
                "device",
            }:
                return None
            start = arguments["0"].constant if len(node.args) == 2 else 0
            end = arguments[str(len(node.args) - 1)]
            dtype = arguments.get("dtype", StaticValue(constant=torch.int64)).constant
            # Only the zero-based integer position range is covered. A floating
            # dtype, unknown start/step, or arbitrary string is not a position axis.
            size = end.constant if type(end.constant) is int else end.symbol
            if (
                type(start) is int
                and start == 0
                and dtype is torch.int64
                and (
                    type(size) is int
                    and 0 <= size <= 1_000_000
                    or type(size) is str
                    and size in self.positive_symbols
                )
            ):
                return self.event(
                    node,
                    None,
                    "tensor_creation",
                    "Integer range",
                    arguments,
                    {
                        "method": "arange",
                        "callable": "torch.arange",
                        "runtime_verified": False,
                        "contract": "torch_zero_based_integer_range.v1",
                        "output_port_specs": {
                            "output": self.spec_data(TensorSpec([size], "I64", "position_range"))
                        },
                    },
                )
            return None
        if function in (torch.ones, torch.zeros) and {key for key in arguments if not key.isdecimal()} <= {
            "dtype",
            "device",
        }:
            dimensions = [arguments[str(index)] for index in range(len(node.args))]
            if len(dimensions) == 1 and dimensions[0].items is not None:
                dimensions = dimensions[0].items
            shape = [value.constant if type(value.constant) is int else value.symbol for value in dimensions]
            dtype = arguments.get("dtype", StaticValue(constant=torch.get_default_dtype())).constant
            if (
                shape
                and all(type(value) is int and value >= 0 or type(value) is str for value in shape)
                and isinstance(dtype, torch.dtype)
                and dtype in DTYPES
            ):
                method = "ones" if function is torch.ones else "zeros"
                return self.event(
                    node,
                    None,
                    "tensor_creation",
                    method.title(),
                    arguments,
                    {
                        "method": method,
                        "callable": f"torch.{method}",
                        "runtime_verified": False,
                        "contract": "torch_static_creation.v1",
                        "output_port_specs": {
                            "output": self.spec_data(TensorSpec(shape, DTYPES[dtype], "created_tensor"))
                        },
                    },
                )
        fields = None
        method = None
        for candidate, name, parameters in (
            (torch.isinf, "isinf", [("input", UNKNOWN)]),
            (torch.clamp, "clamp", [("input", UNKNOWN), ("min", None), ("max", None)]),
            (torch.where, "where", [("condition", UNKNOWN), ("input", UNKNOWN), ("other", UNKNOWN)]),
            (torch.chunk, "chunk", [("input", UNKNOWN), ("chunks", UNKNOWN), ("dim", 0)]),
            (torch.any, "any", [("input", UNKNOWN), ("dim", None), ("keepdim", False)]),
        ):
            if function is candidate:
                fields, method = parameters, name
                break
        if fields is None:
            return None
        bound = self.bind_arguments(node, arguments, fields)
        if bound is None:
            return None
        tensor_names = {"condition", "input", "other"} if method == "where" else {"input"}
        values = {name: bound[name] for name in tensor_names}
        parameters = {name: value.constant for name, value in bound.items() if name not in tensor_names}
        return self.tensor_event(node, method, values, parameters, f"torch.{method}")

    def assign(self, target: ast.AST, value: StaticValue) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            base = self.env.get(name, StaticValue())
            from .native_metadata_contracts import trusted_torch_tensor_method

            full_slice = False
            if (
                self.trusted_t5_tensor_source()
                and trusted_torch_tensor_method("__setitem__")
                and base.spec is not None
                and value.spec is not None
                and base.spec.shape is not None
                and base.spec.shape == value.spec.shape
                and base.spec.dtype == value.spec.dtype
            ):
                axes = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
                if len(axes) == len(base.spec.shape) and all(isinstance(axis, ast.Slice) for axis in axes):
                    full_slice = True
                    for dimension, axis in zip(base.spec.shape, axes):
                        start = self.expression(axis.lower).constant if axis.lower is not None else None
                        stop_value = self.expression(axis.upper) if axis.upper is not None else StaticValue(constant=None)
                        stop = stop_value.constant if stop_value.constant is not UNKNOWN else stop_value.symbol if stop_value.symbol is not None else UNKNOWN
                        step = self.expression(axis.step).constant if axis.step is not None else None
                        if (
                            start not in (None, 0)
                            or type(start) is bool
                            or step not in (None, 1)
                            or type(step) is bool
                            or stop is not None and (type(stop) not in {int, str} or stop != dimension)
                        ):
                            full_slice = False
                            break
            if full_slice:
                self.env[name] = value
            else:
                self.flow.warnings.append(f"Tensor slice assignment to {name} is unresolved; facts invalidated")
                self.env[name] = combine([base, value])
        elif isinstance(target, (ast.Tuple, ast.List)):
            if value.items is not None and len(target.elts) != len(value.items):
                self.flow.warnings.append("Tuple unpacking arity disagrees with the proven source outputs")
                value = combine([value])
            for index, item in enumerate(target.elts):
                self.assign(
                    item,
                    value.items[index]
                    if value.items and index < len(value.items)
                    else StaticValue(
                        [(ref, str(index)) for ref, _ in value.refs],
                        spec=(value.port_specs or {}).get(str(index)),
                    ),
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
                    midpoint = len(self.flow.events)
                    self.env = dict(before)
                    right_returns = self.statements(node.orelse)
                    assignments_changed = before != left or before != self.env
                    if left_returns != right_returns:
                        self.env = self.env if left_returns else left
                    else:
                        for name in left.keys() | self.env.keys():
                            a, b = left.get(name, StaticValue()), self.env.get(name, StaticValue())
                            self.env[name] = a if a == b else join_values([a, b])
                    for event in self.flow.events[start:midpoint]:
                        self.mark_conditional(event, ast.unparse(node.test))
                    for event in self.flow.events[midpoint:]:
                        self.mark_conditional(event, f"not ({ast.unparse(node.test)})")
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
                if (
                    iterable.items is not None
                    and len(iterable.items) <= MAX_STATIC_ITERATIONS
                    and self.iterations + len(iterable.items) <= MAX_STATIC_ITERATIONS
                    and not any(isinstance(item, (ast.Break, ast.Continue)) for item in ast.walk(node))
                ):
                    self.iterations += len(iterable.items)
                    for value in iterable.items:
                        self.assign(node.target, value)
                        if self.statements(node.body):
                            return True
                    if self.statements(node.orelse):
                        return True
                elif iterable.module_path is not None:
                    target = node.target.elts[-1] if isinstance(node.target, ast.Tuple) else node.target
                    before = dict(self.env)
                    self.assign(
                        target,
                        StaticValue(module_path=iterable.module_path, category="container_member"),
                    )
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
            elif isinstance(node, ast.Try):
                unmodeled = []
                if node.handlers:
                    unmodeled.append("exception handlers")
                if node.orelse:
                    unmodeled.append("else path")
                if node.finalbody:
                    unmodeled.append("finally block")
                if unmodeled:
                    self.flow.warnings.append(
                        f"Try control flow is partially modeled at line {node.lineno}; "
                        f"unmodeled: {', '.join(unmodeled)}"
                    )
                if self.statements(node.body):
                    return True
            elif isinstance(node, ast.With):
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
        if self.trusted_t5_tensor_source():
            from .native_metadata_contracts import trusted_torch_tensor_method

            if (
                isinstance(operator, ast.Add)
                and left.spec is not None
                and right.spec is not None
                and trusted_torch_tensor_method("__add__")
            ):
                result = self.tensor_event(
                    node, "float_int_add", {"left": left, "right": right}, {}, "torch.Tensor.__add__"
                )
                if result is not None:
                    return result
            if (
                isinstance(operator, ast.Mult)
                and left.spec is not None
                and right.spec is not None
                and trusted_torch_tensor_method("__imul__" if isinstance(node, ast.AugAssign) else "__mul__")
            ):
                result = self.tensor_event(
                    node, "float_bool_mul", {"left": left, "right": right}, {}, "torch.Tensor.__mul__"
                )
                if result is not None:
                    return result
            if (
                isinstance(operator, ast.Sub)
                and right.spec is not None
                and type(left.constant) in {int, float}
                and trusted_torch_tensor_method("__rsub__")
            ):
                result = self.tensor_event(
                    node, "scalar_left_sub", {"input": right},
                    {"scalar": left.constant}, "torch.Tensor.__rsub__",
                )
                if result is not None:
                    return result
                self.flow.warnings.append("Native T5 scalar-left tensor subtraction has no proven output contract")
        if (
            isinstance(operator, ast.Add)
            and left.items is not None
            and right.items is not None
            and left.category == right.category
            and left.category in {"tuple", "list"}
            and len(left.items) + len(right.items) <= MAX_STATIC_ITERATIONS
        ):
            result = combine([left, right])
            result.items, result.category = [*left.items, *right.items], left.category
            return result
        if not left.refs and not right.refs:
            if isinstance(operator, ast.Add):
                if left.symbol is not None and right.constant == 0:
                    return StaticValue(symbol=left.symbol)
                if right.symbol is not None and left.constant == 0:
                    return StaticValue(symbol=right.symbol)
            if isinstance(operator, ast.Sub) and left.symbol is not None and right.constant == 0:
                return StaticValue(symbol=left.symbol)
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
                    node,
                    None,
                    "scale",
                    "Scale hidden states",
                    {"input": left},
                    {
                        "factor": factor,
                        **(
                            {"output_port_specs": {"output": self.spec_data(left.spec)}}
                            if left.spec is not None and left.spec.dtype in {"F16", "BF16", "F32", "F64"}
                            else {}
                        ),
                    },
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
        if (
            left.spec is not None
            and right.spec is not None
            and left.spec.dtype == right.spec.dtype
            and left.spec.dtype in {"F16", "BF16", "F32", "F64"}
            and isinstance(operator, (ast.Add, ast.Sub, ast.Mult, ast.Div))
        ):
            a, b = left.spec.shape, right.spec.shape
            shape = None
            if a is not None and b is not None:
                rank = max(len(a), len(b))
                a, b = [1] * (rank - len(a)) + a, [1] * (rank - len(b)) + b
                if all(x == y or x == 1 or y == 1 for x, y in zip(a, b)):
                    shape = [y if x == 1 else x for x, y in zip(a, b)]
            metadata["output_port_specs"] = {
                "output": self.spec_data(TensorSpec(shape, left.spec.dtype, "binary_tensor_output"))
            }
            metadata["contract"] = "torch_floating_binary_broadcast.v1"
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
        if name == "dtype" and value.spec is not None and value.spec.dtype is not None:
            dtype = next((key for key, code in DTYPES.items() if code == value.spec.dtype), UNKNOWN)
            return StaticValue(constant=dtype)
        if name == "shape" and value.spec is not None and value.spec.shape is not None:
            return StaticValue(
                items=[self.dimension_value(size) for size in value.spec.shape], category="tuple"
            )
        if name == "device" and value.spec is not None:
            return StaticValue(category="torch_device")
        if name == "ndim" and value.spec is not None and value.spec.shape is not None:
            return StaticValue(constant=len(value.spec.shape))
        if name in {"shape", "dtype", "device", "ndim"}:
            if name == "dtype" and value.module_path is not None:
                from .native_metadata_contracts import native_module_attribute

                module = self.model.get_submodule(value.module_path) if value.module_path else self.model
                supported, native_value = native_module_attribute(module, name)
                if supported:
                    diagnostic = {
                        "code": "native_module_attribute",
                        "attribute": name,
                        "module_class": f"{type(module).__module__}.{type(module).__name__}",
                        "basis": "Source-fingerprinted native dtype property; homogeneous meta parameter metadata",
                    }
                    if diagnostic not in self.flow.diagnostics:
                        self.flow.diagnostics.append(diagnostic)
                    return StaticValue(constant=native_value)
            return StaticValue()
        if isinstance(value.constant, ModuleType):
            attribute = vars(value.constant).get(name, UNKNOWN)
            if (
                isinstance(attribute, (ModuleType, torch.dtype))
                or inspect.isfunction(attribute)
                or inspect.isbuiltin(attribute)
            ):
                return StaticValue(constant=attribute)
            return self.primitive(attribute)
        if value.fields is not None and name in value.fields:
            return value.fields[name]
        if value.module_path is not None:
            path = f"{value.module_path}.{name}" if value.module_path else name
            if path in self.classifications:
                return StaticValue(module_path=path)
            module = self.model.get_submodule(value.module_path) if value.module_path else self.model
            attribute = vars(module).get(name, UNKNOWN)
            if attribute is UNKNOWN:
                for members in (module._parameters, module._buffers, module._modules):
                    if name in members:
                        attribute = members[name]
                        break
            if attribute is UNKNOWN:
                descriptor = inspect.getattr_static(type(module), name, UNKNOWN)
                if isinstance(descriptor, (staticmethod, classmethod)) or inspect.isfunction(descriptor):
                    return StaticValue(module_path=value.module_path, method=name)
                if isinstance(descriptor, property) or hasattr(type(descriptor), "__get__"):
                    if name == "dtype":
                        from .native_metadata_contracts import native_module_attribute

                        supported, native_value = native_module_attribute(module, name)
                        if supported:
                            diagnostic = {
                                "code": "native_module_attribute",
                                "attribute": name,
                                "module_class": f"{type(module).__module__}.{type(module).__name__}",
                                "basis": "Source-fingerprinted native dtype property; homogeneous meta parameter metadata",
                            }
                            if diagnostic not in self.flow.diagnostics:
                                self.flow.diagnostics.append(diagnostic)
                            return StaticValue(constant=native_value)
                    return StaticValue()
                attribute = descriptor
            if isinstance(attribute, torch.Tensor):
                return StaticValue(
                    [(f"parameter:{path}", "output")],
                    spec=TensorSpec(
                        list(attribute.shape), DTYPES.get(attribute.dtype), "native_tensor_metadata"
                    ),
                )
            if inspect.ismethod(attribute) or inspect.isfunction(attribute):
                return StaticValue(module_path=value.module_path, method=name)
            return StaticValue(constant=attribute)
        if value.constant is not UNKNOWN and value.constant is not None:
            attribute = (
                vars(value.constant).get(name, UNKNOWN) if hasattr(value.constant, "__dict__") else UNKNOWN
            )
            if attribute is UNKNOWN:
                attribute = inspect.getattr_static(type(value.constant), name, UNKNOWN)
                if isinstance(attribute, property) or hasattr(type(attribute), "__get__"):
                    from .native_metadata_contracts import native_config_attribute

                    supported, native_value = native_config_attribute(value.constant, name)
                    if supported:
                        diagnostic = {
                            "code": "native_config_attribute",
                            "attribute": name,
                            "basis": "Source-fingerprinted native descriptor backing field; no descriptor invocation",
                        }
                        if diagnostic not in self.flow.diagnostics:
                            self.flow.diagnostics.append(diagnostic)
                        return StaticValue(constant=native_value)
                    return StaticValue()
            return StaticValue(constant=attribute)
        return StaticValue([(ref, name) for ref, _ in value.refs])

    def source_callable(self, expression: ast.AST) -> Any:
        """Resolve globals through module dictionaries without invoking descriptors or model code."""
        parts = []
        while isinstance(expression, ast.Attribute):
            parts.append(expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            return UNKNOWN
        if expression.id in self.env:
            function = self.env[expression.id].constant
        else:
            function = self.source_globals.get(expression.id, vars(builtins).get(expression.id, UNKNOWN))
        for name in reversed(parts):
            if not isinstance(function, ModuleType):
                return UNKNOWN
            function = vars(function).get(name, UNKNOWN)
        return function

    def functional_operation(self, node: ast.Call, arguments: dict[str, StaticValue]) -> StaticValue | None:
        """Describe a bounded set of actual Torch operators; similarly named helpers stay opaque."""
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            return None
        function = self.source_callable(node.func)
        contracts = (
            (
                torch.nn.functional.dropout,
                "dropout",
                "Dropout",
                {"p": 0.5, "training": True, "inplace": False},
            ),
            (torch.nn.functional.relu, "activation", "ReLU", {"inplace": False}),
            (torch.nn.functional.silu, "activation", "SiLU", {"inplace": False}),
            (torch.nn.functional.gelu, "activation", "GELU", {"approximate": "none"}),
            (torch.tanh, "activation", "Tanh", {}),
            (torch.sigmoid, "activation", "Sigmoid", {}),
            (
                torch.nn.functional.softmax,
                "softmax",
                "Softmax",
                {"dim": None, "_stacklevel": 3, "dtype": None},
            ),
        )
        contract = next((item for item in contracts if function is item[0]), None)
        if contract is None:
            return None
        _, kind, label, defaults = contract
        parameters = [inspect.Parameter("input", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        for name, default in defaults.items():
            parameter_kind = (
                inspect.Parameter.KEYWORD_ONLY
                if function is torch.nn.functional.gelu
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
            parameters.append(inspect.Parameter(name, parameter_kind, default=StaticValue(constant=default)))
        try:
            bound = inspect.Signature(parameters).bind(
                *[arguments[str(index)] for index in range(len(node.args))],
                **{key: value for key, value in arguments.items() if not key.isdecimal()},
            )
            bound.apply_defaults()
        except TypeError:
            return None
        input_value = bound.arguments["input"]
        if not input_value.refs or kind == "softmax" and bound.arguments["dtype"].constant is not None:
            return None
        parameters = {
            name: value.constant
            if type(value.constant) in (type(None), bool, int, str)
            or type(value.constant) is float
            and math.isfinite(value.constant)
            else {"known": False}
            for name, value in bound.arguments.items()
            if name != "input"
        }
        if any(isinstance(value, dict) for value in parameters.values()):
            return None
        if function is torch.nn.functional.dropout and (
            type(parameters["p"]) not in (int, float)
            or not 0 <= parameters["p"] <= 1
            or type(parameters["training"]) is not bool
            or type(parameters["inplace"]) is not bool
        ):
            return None
        if (
            function in (torch.nn.functional.relu, torch.nn.functional.silu)
            and type(parameters["inplace"]) is not bool
        ):
            return None
        if function is torch.nn.functional.gelu and parameters["approximate"] not in {"none", "tanh"}:
            return None
        if function is torch.nn.functional.softmax:
            dim = parameters["dim"]
            shape = input_value.spec.shape if input_value.spec is not None else None
            if dim is not None and (
                type(dim) is not int or shape is not None and not -len(shape) <= dim < len(shape)
            ):
                return None
        metadata = {
            "callable": f"{function.__module__}.{function.__qualname__}",
            "method": function.__name__,
            "parameters": parameters,
            "shape_basis": "Torch unary shape-preserving operator contract",
            "torch_version": str(torch.__version__),
            "runtime_verified": False,
        }
        if input_value.spec is not None and input_value.spec.dtype in {"F16", "BF16", "F32", "F64"}:
            metadata["output_port_specs"] = {"output": self.spec_data(input_value.spec)}
        return self.event(node, None, kind, label, {"input": input_value}, metadata)

    def canonical_predicate(self, node: ast.Call, arguments: dict[str, StaticValue]) -> StaticValue | None:
        """Resolve native predicates by identity, never by the spelling of a call."""
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return None
        function = self.source_callable(node.func)
        if function is isinstance:
            if len(node.args) != 2 or node.keywords:
                return None
            target = self.source_callable(node.args[1])
            value = arguments["0"].constant
            if target is torch.Tensor:
                spec = arguments["0"].spec
                if spec is not None and (spec.shape is not None or spec.dtype is not None):
                    return StaticValue(constant=True)
                if type(value) in (type(None), bool, int, float, str):
                    return StaticValue(constant=False)
            if type(target) is type and type(value) in (type(None), bool, int, float, str):
                return StaticValue(constant=target in type(value).__mro__)
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
            if node.id in self.env:
                return self.env[node.id]
            value = self.source_globals.get(node.id, UNKNOWN)
            if (
                isinstance(value, (ModuleType, torch.dtype))
                or inspect.isfunction(value)
                or inspect.isbuiltin(value)
            ):
                return StaticValue(constant=value)
            return self.primitive(value)
        if isinstance(node, ast.Starred):
            return self.expression(node.value)
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
            result.category = "tuple" if isinstance(node, ast.Tuple) else "list"
            return result
        if isinstance(node, ast.Dict):
            keys = [self.expression(key) for key in node.keys]
            values = [self.expression(value) for value in node.values]
            result = combine([*keys, *values])
            if all(type(key.constant) is str for key in keys):
                result.fields = {key.constant: value for key, value in zip(keys, values)}
            return result
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                value = self.expression(node.value)
                bounds = [
                    self.expression(item).constant if item is not None else None
                    for item in (node.slice.lower, node.slice.upper, node.slice.step)
                ]
                if (
                    value.items is not None
                    and all(item is None or type(item) is int for item in bounds)
                    and bounds[2] != 0
                ):
                    items = value.items[slice(*bounds)]
                    result = combine(items)
                    result.items, result.category = items, value.category
                    return result
            if isinstance(node.slice, (ast.Tuple, ast.Slice, ast.Constant)) and self.trusted_t5_tensor_source():
                from .native_metadata_contracts import trusted_torch_tensor_method

                value = self.expression(node.value)
                if (
                    trusted_torch_tensor_method("__getitem__")
                    and value.spec is not None
                    and value.spec.shape is not None
                ):
                    raw_indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                    indices = []
                    supported = True
                    for item in raw_indices:
                        if isinstance(item, ast.Constant) and item.value is None:
                            indices.append(None)
                        elif isinstance(item, ast.Slice):
                            start = self.expression(item.lower) if item.lower is not None else StaticValue(constant=None)
                            stop = self.expression(item.upper) if item.upper is not None else StaticValue(constant=None)
                            step = self.expression(item.step) if item.step is not None else StaticValue(constant=None)
                            start_value = start.constant if start.constant is not UNKNOWN else start.symbol if start.symbol is not None else UNKNOWN
                            stop_value = stop.constant if stop.constant is not UNKNOWN else stop.symbol if stop.symbol is not None else UNKNOWN
                            step_value = step.constant if step.constant is not UNKNOWN else step.symbol if step.symbol is not None else UNKNOWN
                            if (
                                start_value is UNKNOWN
                                or stop_value is UNKNOWN
                                or step_value is UNKNOWN
                                or
                                start_value is not None and type(start_value) is not int
                                or stop_value is not None and type(stop_value) not in (int, str)
                                or step_value is not None and type(step_value) is not int
                            ):
                                supported = False
                                break
                            indices.append({"start": start_value, "stop": stop_value, "step": step_value})
                        else:
                            supported = False
                            break
                    if supported:
                        return self.tensor_event(
                            node,
                            "slice",
                            {"input": value},
                            {"indices": indices},
                            "torch.Tensor.__getitem__",
                        )
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
            if type(index) is int and value.port_specs and str(index) in value.port_specs:
                return StaticValue(
                    [(ref, str(index)) for ref, _ in value.refs], spec=value.port_specs[str(index)]
                )
            if value.fields and type(index) is str and index in value.fields:
                return value.fields[index]
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
            candidates, uncertain = [], False
            for item in node.values:
                start = len(self.flow.events)
                value = self.expression(item)
                if uncertain:
                    for event in self.flow.events[start:]:
                        self.mark_conditional(event, ast.unparse(node))
                    if len(self.flow.events) > start:
                        self.flow.warnings.append(
                            "Conditional source path is unresolved: " + ast.unparse(node)
                        )
                known = type(value.constant) in (type(None), bool, int, float, str)
                if not known:
                    candidates.append(value)
                    uncertain = True
                elif bool(value.constant) != isinstance(node.op, ast.And):
                    return join_values([*candidates, value]) if uncertain else value
            return join_values([*candidates, value]) if uncertain else value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.expression(node.left), self.expression(node.comparators[0])
            if self.trusted_t5_tensor_source():
                from .native_metadata_contracts import trusted_torch_tensor_method

                if (
                    isinstance(node.ops[0], ast.Gt)
                    and left.spec is not None
                    and right.spec is not None
                    and trusted_torch_tensor_method("__gt__")
                ):
                    result = self.tensor_event(
                        node, "gt", {"left": left, "right": right}, {}, "torch.Tensor.__gt__"
                    )
                    if result is not None:
                        return result
                if isinstance(node.ops[0], ast.Eq) and trusted_torch_tensor_method("__eq__"):
                    if left.spec is not None and right.spec is not None:
                        result = self.tensor_event(
                            node, "eq", {"left": left, "right": right}, {}, "torch.Tensor.__eq__"
                        )
                        if result is not None:
                            return result
                    if left.spec is not None and type(right.constant) in {int, float}:
                        result = self.tensor_event(
                            node, "eq_scalar", {"input": left},
                            {"scalar": right.constant}, "torch.Tensor.__eq__",
                        )
                        if result is not None:
                            return result
            if (
                isinstance(left.constant, torch.dtype)
                and isinstance(right.constant, torch.dtype)
                and isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Is, ast.IsNot))
            ):
                same = left.constant is right.constant
                return StaticValue(
                    constant=not same if isinstance(node.ops[0], (ast.NotEq, ast.IsNot)) else same
                )
            if (
                isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and type(left.constant) in (type(None), bool)
                and type(right.constant) in (type(None), bool)
            ):
                same = left.constant is right.constant
                return StaticValue(constant=same if isinstance(node.ops[0], ast.Is) else not same)
            if isinstance(node.ops[0], (ast.Is, ast.IsNot)) and right.constant is None:
                if (
                    left.spec is not None
                    and (left.spec.shape is not None or left.spec.dtype is not None)
                    or left.module_path is not None
                    or left.items is not None
                    or left.fields is not None
                ):
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
                if type(left.constant) is int and type(right.constant) is int:
                    operations = {
                        ast.Lt: arithmetic.lt,
                        ast.LtE: arithmetic.le,
                        ast.Gt: arithmetic.gt,
                        ast.GtE: arithmetic.ge,
                    }
                    if type(node.ops[0]) in operations:
                        return StaticValue(
                            constant=operations[type(node.ops[0])](left.constant, right.constant)
                        )
            return StaticValue()
        if isinstance(node, ast.IfExp):
            condition = self.expression(node.test).constant
            if condition is not UNKNOWN:
                return self.expression(node.body if condition else node.orelse)
            before, start = dict(self.env), len(self.flow.events)
            left = self.expression(node.body)
            left_env = dict(self.env)
            midpoint = len(self.flow.events)
            self.env = dict(before)
            right = self.expression(node.orelse)
            self.env = {
                name: join_values([left_env.get(name, StaticValue()), self.env.get(name, StaticValue())])
                for name in left_env.keys() | self.env.keys()
            }
            for event in self.flow.events[start:midpoint]:
                self.mark_conditional(event, ast.unparse(node.test))
            for event in self.flow.events[midpoint:]:
                self.mark_conditional(event, f"not ({ast.unparse(node.test)})")
            self.flow.warnings.append("Conditional source path is unresolved: " + ast.unparse(node.test))
            return join_values([left, right])
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
            arguments = {str(i): self.expression(arg) for i, arg in enumerate(node.args)}
            arguments.update(
                {
                    kw.arg if kw.arg is not None else f"**{index}": self.expression(kw.value)
                    for index, kw in enumerate(node.keywords)
                }
            )
            builtin = self.builtin_metadata(node, arguments)
            if builtin is not None:
                return builtin
            if receiver is not None:
                result = self.tensor_method(node, receiver, arguments)
                if result is not None:
                    return result
            if (
                isinstance(node.func, ast.Attribute)
                and not isinstance(receiver.constant, ModuleType)
                and name in {"size", "dim", "numel"}
            ):
                return StaticValue()
            if (
                isinstance(node.func, ast.Attribute)
                and not isinstance(receiver.constant, ModuleType)
                and name
                in {
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
                    "clone",
                    "masked_fill",
                }
            ):
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
                    values = [arguments[str(index)].constant for index in range(len(node.args))]
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
                self.flow.warnings.append(f"Tensor transform {name} has no proven output contract")
                dtype = (
                    receiver.spec.dtype
                    if receiver.spec is not None and name not in {"to", "type_as", "float", "half"}
                    else None
                )
                return self.event(
                    node,
                    None,
                    "tensor_operation",
                    name.title(),
                    {"input": receiver},
                    {
                        "method": name,
                        "runtime_verified": False,
                        "output_port_specs": {
                            "output": self.spec_data(TensorSpec(None, dtype, "unresolved_tensor_transform"))
                        },
                    },
                )
            predicate = self.canonical_predicate(node, arguments)
            if predicate is not None:
                return predicate
            tensor = self.torch_operation(node, arguments)
            if tensor is not None:
                return tensor
            functional = self.functional_operation(node, arguments)
            if functional is not None:
                return functional
            if function.module_path is not None and function.method is None:
                return self.event(
                    node,
                    function.module_path,
                    "module",
                    function.module_path,
                    arguments,
                    native_container_member=function.category == "container_member",
                )
            if function.method is not None and self.depth < 3:
                return self.helper(
                    function.module_path,
                    function.method,
                    [arguments[str(index)] for index in range(len(node.args))],
                    node,
                    keywords={key: value for key, value in arguments.items() if not key.isdecimal()},
                )
            if self.source_callable(node.func) is apply_chunking_to_forward:
                if (
                    len(node.args) >= 4
                    and not node.keywords
                    and not any(isinstance(arg, ast.Starred) for arg in node.args)
                    and type(arguments["1"].constant) is int
                    and arguments["1"].constant == 0
                ):
                    helper = arguments["0"]
                    if helper.method:
                        return self.helper(
                            helper.module_path,
                            helper.method,
                            [arguments[str(index)] for index in range(3, len(node.args))],
                            node,
                        )
                self.flow.warnings.append("Chunked helper arguments are unresolved; dependencies only")
                return combine(list(arguments.values()))
            if name.startswith("create_") and name.endswith("mask"):
                return StaticValue()
            if name in {"matmul", "bmm", "einsum", "cat", "stack"} and self.source_callable(
                node.func
            ) is getattr(torch, name):
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
                        present = attribute in vars(module) or any(
                            attribute in members
                            for members in (module._parameters, module._buffers, module._modules)
                        )
                        if present or inspect.getattr_static(type(module), attribute, UNKNOWN) is not UNKNOWN:
                            return StaticValue(constant=True)
                        return StaticValue(constant=False)
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
        self,
        path: str | None,
        method_name: str,
        values: list[StaticValue],
        node: ast.AST,
        *,
        keywords: dict[str, StaticValue] | None = None,
    ) -> StaticValue:
        keywords = keywords or {}
        dependencies = [*values, *keywords.values()]
        module = self.model.get_submodule(path) if path else self.model
        try:
            descriptor = inspect.getattr_static(type(module), method_name)
            method = inspect.unwrap(
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else getattr(type(module), method_name)
            )
            function = next(
                n
                for n in ast.parse(textwrap.dedent(inspect.getsource(method))).body
                if isinstance(n, ast.FunctionDef)
            )
        except (AttributeError, OSError, TypeError, SyntaxError, StopIteration):
            self.flow.warnings.append(f"Helper source unavailable: {path}.{method_name}; dependencies only")
            return combine(dependencies)
        if isinstance(node, ast.Call) and (
            any(isinstance(arg, ast.Starred) for arg in node.args)
            or any(keyword.arg is None for keyword in node.keywords)
        ):
            self.flow.warnings.append(f"Helper argument expansion is unresolved: {path}.{method_name}")
            return combine(dependencies)
        try:
            receiver = (
                []
                if isinstance(descriptor, staticmethod)
                else [StaticValue() if isinstance(descriptor, classmethod) else StaticValue(module_path=path)]
            )
            bound = inspect.signature(method).bind(*receiver, *values, **keywords)
            bound.apply_defaults()
        except (ValueError, TypeError):
            self.flow.warnings.append(
                f"Helper arguments cannot be bound: {path}.{method_name}; dependencies only"
            )
            return combine(dependencies)
        environment = {}
        for name, value in bound.arguments.items():
            if isinstance(value, StaticValue):
                environment[name] = value
            elif type(value) in (type(None), bool, int, str) or type(value) is float and math.isfinite(value):
                environment[name] = StaticValue(constant=value)
            elif type(value) is tuple and all(isinstance(item, StaticValue) for item in value):
                environment[name] = combine(list(value))
                environment[name].items = list(value)
            elif type(value) is dict and all(isinstance(item, StaticValue) for item in value.values()):
                environment[name] = combine(list(value.values()))
                environment[name].fields = dict(value)
            else:
                environment[name] = StaticValue()
        old_env, old_output, old_globals, old_returned, old_method_name, old_method = (
            self.env,
            self.flow.outputs,
            self.source_globals,
            self.returned,
            self.source_method_name,
            self.source_method,
        )
        self.source_globals = method.__globals__
        self.source_method_name, self.source_method = method_name, method
        self.returned = []
        self.env = environment
        self.depth += 1
        try:
            self.statements(function.body)
            return merge_outputs(self.returned)
        finally:
            self.depth -= 1
            self.env, self.flow.outputs = old_env, old_output
            self.source_globals = old_globals
            self.returned = old_returned
            self.source_method_name, self.source_method = old_method_name, old_method
