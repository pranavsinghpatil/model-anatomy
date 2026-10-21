"""Generic semantic hierarchy assembled from static source dependencies and patterns."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

import torch

from .inputs import InputResolver
from .ir import Edge, Evidence, Graph, Node, RepetitionSpec, TensorSpec
from .patterns import Repetition, SemanticAnalyzer
from .shape_ops import operation_shape
from .static_flow import FlowEvent, StaticFlowAnalyzer

LABELS = {
    "embedding": "Token Embedding",
    "position_embedding": "Position Embedding",
    "embeddings": "Embeddings",
    "patch_embedding": "Patch Embedding",
    "rotary_position_embedding": "Rotary Position Embedding",
    "self_attention": "Self Attention",
    "cross_attention": "Cross Attention",
    "attention_core": "Attention Core",
    "feed_forward": "Feed Forward / MLP",
    "gated_mlp": "Gated MLP",
    "layer_norm": "LayerNorm",
    "rms_norm": "RMSNorm",
    "batch_norm": "BatchNorm",
    "normalization": "Normalization",
    "dropout": "Dropout",
    "projection": "Projection",
    "decoder_block": "Decoder Block",
    "encoder_block": "Encoder Block",
    "transformer_block": "Transformer Block",
    "residual_block": "Residual Block",
    "residual_stage": "Residual Stage",
    "convolution": "Convolution",
    "pooling": "Pooling",
    "activation": "Activation",
    "decoder": "Decoder",
    "encoder": "Encoder",
    "backbone": "Backbone",
    "lm_head": "LM Head",
    "classifier": "Classifier",
    "stem": "Stem",
    "expert": "Expert",
    "expert_group": "Expert Group",
    "router": "Router",
    "container": "Component Group",
    "convolution_block": "Convolution / Norm",
    "projection_block": "Projection / Norm",
}


def _output_port_specs(metadata: Mapping[str, Any]) -> dict[str, TensorSpec] | None:
    """Return only complete, schema-valid contracts supplied by static analysis."""
    if "output_port_specs" not in metadata:
        return None
    raw = metadata["output_port_specs"]
    if not isinstance(raw, Mapping) or not raw:
        return None
    result: dict[str, TensorSpec] = {}
    for port, value in raw.items():
        if not isinstance(port, str) or not port or not isinstance(value, Mapping):
            return None
        if set(value) != {"shape", "dtype", "semantic_name"}:
            return None
        shape, dtype, name = value["shape"], value["dtype"], value["semantic_name"]
        if shape is not None and (
            not isinstance(shape, list)
            or any(
                not (
                    (type(dim) is int and dim >= 0)
                    or (isinstance(dim, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", dim))
                )
                for dim in shape
            )
        ):
            return None
        if dtype is not None and (
            not isinstance(dtype, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", dtype)
        ):
            return None
        if name is not None and (not isinstance(name, str) or not name.strip()):
            return None
        result[port] = TensorSpec(shape, dtype, name if name is not None else f"output:{port}")
    return result


def identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value) or "model"


class HierarchyBuilder:
    def __init__(self, model: torch.nn.Module, raw: dict[str, Any], analyzer: SemanticAnalyzer):
        self.model, self.raw, self.analyzer = model, raw, analyzer
        self.inputs = InputResolver(model)
        self.graph_ids: set[str] = set()
        self.flow_reports: dict[str, dict[str, Any]] = {}
        self.source_sequential_containers: dict[str, dict[str, Any]] = {}
        self.active: set[str] = set()

    def graph_id(self, path: str, suggested: str | None = None) -> str:
        if not path:
            suggested = "overview"
        if self.analyzer.adapter and suggested is None:
            suggested = self.analyzer.adapter.graph_aliases.get(self.raw["modules"].get(path, {}).get("type"))
        candidate = suggested or identifier(path)
        base, index = candidate, 1
        while candidate in self.graph_ids:
            index += 1
            candidate = f"{base}_{index}"
        self.graph_ids.add(candidate)
        return candidate

    def evidence(self, path: str, rule: str, kind: str = "reconstructed") -> Evidence:
        return Evidence(
            "static_semantic_analysis",
            kind,
            rule,
            {
                "module_path": path,
                "source_fingerprint": self.raw["source_fingerprints"].get(path),
                "runtime_verified": False,
            },
        )

    def module_node(
        self, path: str, node_id: str, repeat: Repetition | None = None, build_child: bool = True
    ) -> Node:
        record, classification = self.raw["modules"][path], self.analyzer.classifications[path]
        paths = repeat.instance_paths if repeat else [path]
        records = [
            item
            for name, item in self.raw["parameters"].items()
            if any(name.startswith(p + ".") for p in paths)
        ]
        unique = {item["canonical_parameter"]: item for item in records}
        metadata = {
            "classification": asdict(classification),
            "parameter_refs": [p["name"] for p in records],
            "parameter_shapes": {p["name"]: p["shape"] for p in records},
            "attributes": record["attributes"],
            "structural_signature": self.analyzer.signatures[path].digest,
            "parameter_count_scope": "unique module parameter identities within this component; hierarchy and shared aliases overlap",
        }
        label = LABELS.get(classification.kind, record["type"])
        if classification.kind == "backbone":
            label = self.model.config.model_type.upper() + " Backbone"
        if path.rpartition(".")[0] == self.model.base_model_prefix and path.rsplit(".", 1)[-1] in {
            "norm",
            "ln_f",
            "final_layer_norm",
            "layernorm",
        }:
            label = "Final " + label
        if self.analyzer.adapter:
            label = self.analyzer.adapter.type_labels.get(record["type"], label)
        node = Node(
            node_id,
            label,
            classification.kind,
            path,
            record["type"],
            parameter_count=sum(p["parameter_count"] for p in unique.values()),
            metadata=metadata,
            evidence=[
                Evidence("named_modules", "direct", path),
                self.evidence(path, "semantic_classification"),
            ],
            confidence=classification.confidence,
            source_modules=paths,
        )
        shared = sorted(
            {alias for item in records for alias in item["aliases"] if not alias.startswith(path + ".")}
        )
        if shared:
            node.metadata["shared_with"] = shared[0] if len(shared) == 1 else shared
            node.metadata["additional_unique_parameter_count"] = sum(
                item["parameter_count"]
                for canonical, item in unique.items()
                if canonical.startswith(path + ".")
            )
        if repeat:
            node.repeat = repeat.count
            node.repetition = RepetitionSpec(repeat.count, repeat.mode, paths)
            node.metadata.update(
                instance_paths=paths,
                template_path=path,
                per_instance_parameter_count=self.analyzer.signatures[path].parameter_count,
                repetition_evidence=repeat.evidence,
            )
        if build_child and record["children"] and path not in self.active:
            node.child_graph = self.build(path)
        return node

    def call_node(self, event: FlowEvent, node_id: str) -> Node:
        path = event.module_path
        groups = self.analyzer.repetitions.get(path)
        if groups and len(groups) == 1:
            node = self.module_node(groups[0].instance_paths[0], node_id, groups[0])
            node.module_path, node.module_type = path, self.raw["modules"][path]["type"]
            node.metadata["instance_module_type"] = self.raw["modules"][groups[0].instance_paths[0]]["type"]
        else:
            node = self.module_node(path, node_id)
            if groups and node.kind == "container":
                node.kind, node.label = (
                    "block_stack",
                    f"Block Stack ({sum(g.count for g in groups)} instances)",
                )
                node.metadata["variant_groups"] = [dict(asdict(g), count=g.count) for g in groups]
        return node

    def output_spec(self, node: Node, incoming: TensorSpec | None = None) -> TensorSpec:
        path, kind = node.module_path, node.kind
        if path is None:
            if self.inputs.native_vit and kind in {"reshape", "selection", "concatenation"}:
                return operation_shape(kind, node.metadata, node.inputs)
            if kind in {"matmul", "concatenation", "operation"}:
                return TensorSpec(None, None, "unresolved_operation_output")
            return incoming or TensorSpec(None, None, "result")
        module = self.model.get_submodule(path)
        if self.inputs.native_vit and isinstance(module, torch.nn.Linear):
            shape = list(incoming.shape) if incoming and incoming.shape is not None else None
            if shape and shape[-1] == module.in_features:
                shape[-1] = module.out_features
            else:
                shape = None
            return TensorSpec(shape, self.inputs.activation_dtype, "projected_states")
        if self.inputs.native_vit and isinstance(module, torch.nn.Conv2d):
            shape = incoming.shape if incoming else None
            result = None
            if (
                shape
                and len(shape) == 4
                and shape[1] == module.in_channels
                and all(type(v) is int for v in shape[2:])
                and not isinstance(module.padding, str)
            ):
                spatial = [
                    (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
                    for size, pad, dilation, kernel, stride in zip(
                        shape[2:], module.padding, module.dilation, module.kernel_size, module.stride
                    )
                ]
                if all(v > 0 for v in spatial):
                    result = [shape[0], module.out_channels, *spatial]
            return TensorSpec(result, self.inputs.activation_dtype, "feature_map")
        if kind == "rotary_position_embedding":
            frequency = getattr(module, "inv_freq", None)
            dimension = frequency.numel() * 2 if isinstance(frequency, torch.Tensor) else None
            return TensorSpec(
                ["B", "S", dimension] if dimension else None, self.inputs.activation_dtype, "rotary_cos_sin"
            )
        if isinstance(module, torch.nn.Embedding):
            sequence = "T" if "decoder" in path and self.model.config.is_encoder_decoder else "S"
            return TensorSpec(
                [1 if kind == "position_embedding" else "B", sequence, module.embedding_dim],
                self.inputs.activation_dtype,
                "embedding",
            )
        if kind in {"classifier", "lm_head"}:
            output = getattr(module, "out_features", getattr(self.model.config, "vocab_size", None))
            if kind == "classifier":
                output = getattr(self.model.config, "num_labels", output)
                shape = (
                    ["B", "S", output]
                    if "TokenClassification" in type(self.model).__name__
                    else ["B", output]
                )
            else:
                shape = ["B", "T" if self.model.config.is_encoder_decoder else "S", output]
            return TensorSpec(shape if type(output) is int else None, self.inputs.activation_dtype, "logits")
        if isinstance(module, torch.nn.Linear):
            shape = list(incoming.shape) if incoming and incoming.shape is not None else None
            if shape and shape[-1] == module.in_features:
                shape[-1] = module.out_features
                return TensorSpec(shape, incoming.dtype, "projected_states")
            return TensorSpec(None, None, "unresolved_linear_input")
        if self.raw["modules"][path]["type"] == "Conv1D":
            shape = list(incoming.shape) if incoming and incoming.shape is not None else None
            weight = getattr(module, "weight", None)
            in_features = getattr(module, "nx", None)
            if in_features is None and isinstance(weight, torch.Tensor) and weight.ndim == 2:
                in_features = weight.shape[0]
            if shape and type(in_features) is int and shape[-1] == in_features:
                shape[-1] = module.nf
                return TensorSpec(shape, incoming.dtype, "projected_states")
            return TensorSpec(None, None, "unresolved_conv1d_input")
        if kind in {"convolution", "pooling", "stem", "residual_stage", "residual_block"}:
            return TensorSpec(None, self.inputs.activation_dtype, "feature_map")
        if kind in {
            "dropout",
            "layer_norm",
            "rms_norm",
            "normalization",
            "batch_norm",
            "activation",
            "identity",
        }:
            return incoming if incoming is not None else TensorSpec(None, None, "unresolved_passthrough_input")
        if node.child_graph:
            output_node = next((n for n in node.child_graph.nodes if n.kind == "output"), None)
            if output_node and output_node.outputs:
                child_output = output_node.outputs[0]
                if child_output.shape is not None or child_output.dtype is not None:
                    return child_output
        if self.inputs.native_vit and kind in {"backbone", "encoder"}:
            # ViT's backbone token layout is established by the native config/source path.
            return self.inputs.spec("hidden_states", path)
        return TensorSpec(None, None, "unknown")

    def connect(
        self,
        graph: Graph,
        source: Node,
        target: Node,
        spec: TensorSpec | None = None,
        source_port: str = "output",
        target_port: str = "input",
        label: str | None = None,
    ) -> None:
        if spec is None:
            port_specs = _output_port_specs(source.metadata)
            contract_declared = (
                "output_port_specs" in source.metadata
                or source.metadata.get("output_port_specs_invalid") is True
                or source.metadata.get("output_contract_required") is True
            )
            if port_specs is not None and source_port in port_specs:
                spec = port_specs[source_port]
            elif contract_declared:
                spec = TensorSpec(None, None, f"unresolved_port:{source_port}")
            else:
                primary = source_port in {"output", "last_hidden_state", "logits"}
                known_rotary = source.kind == "rotary_position_embedding" and source_port in {"0", "1"}
                spec = (
                    source.outputs[0]
                    if source.outputs and (primary or known_rotary)
                    else TensorSpec(None, None, f"unresolved_port:{source_port}")
                )
        if spec not in source.outputs:
            source.outputs.append(spec)
        target.inputs.append(spec)
        graph.edges.append(
            Edge(
                source.id,
                target.id,
                spec,
                source_port,
                target_port,
                label,
                [self.evidence(graph.metadata.get("module_path", ""), "static_source_dependency")],
            )
        )

    def build(self, path: str = "") -> Graph:
        self.active.add(path)
        module = self.model.get_submodule(path) if path else self.model
        classification = self.analyzer.classifications[path]
        children = self.raw["modules"][path]["children"]
        names = {p.rsplit(".", 1)[-1].lower() for p in children}
        if classification.kind in {"self_attention", "cross_attention"} and (
            any(
                qkv.issubset(names)
                for qkv in ({"q_proj", "k_proj", "v_proj"}, {"query", "key", "value"}, {"q", "k", "v"})
            )
            or {"c_attn", "c_proj"}.issubset(names)
        ):
            graph = self.attention(path)
        elif path in self.analyzer.repetitions:
            graph = self.container(path)
        elif path in self.source_sequential_containers and type(module) is torch.nn.ModuleList:
            graph = self.sequential(path)
            graph.metadata.update(
                topology="static_source_proven_module_list",
                sequence_evidence=self.source_sequential_containers[path],
            )
        elif isinstance(module, torch.nn.Sequential):
            graph = self.sequential(path)
        else:
            graph = self.source_graph(path)
        self.active.remove(path)
        if not path:
            self.flatten_decoder_backbone(graph)
        return graph

    def sequential(self, path: str) -> Graph:
        graph = Graph(
            self.graph_id(path),
            [],
            [],
            {"module_path": path, "topology": "nn.Sequential contract", "runtime_verified": False},
        )
        current = Node("input", "Input", "input", outputs=[self.inputs.spec("hidden_states", path)])
        graph.nodes.append(current)
        for index, child in enumerate(self.raw["modules"][path]["children"]):
            node = self.module_node(child, f"component{index}")
            self.connect(graph, current, node)
            node.outputs = [self.output_spec(node, current.outputs[0])]
            graph.nodes.append(node)
            current = node
        output = Node("output", "Output", "output", outputs=list(current.outputs))
        graph.nodes.append(output)
        self.connect(graph, current, output)
        return graph

    def container(self, path: str) -> Graph:
        groups = self.analyzer.repetitions[path]
        graph = Graph(
            self.graph_id(path),
            [],
            [],
            {
                "module_path": path,
                "topology": "repeated subtrees",
                "runtime_verified": False,
                "total_instances": sum(g.count for g in groups),
            },
        )
        current = Node("input", "Input", "input", outputs=[self.inputs.spec("hidden_states", path)])
        graph.nodes.append(current)
        for index, group in enumerate(groups):
            node = self.module_node(group.instance_paths[0], f"group{index}", group)
            graph.nodes.append(node)
            node.outputs = [self.output_spec(node)]
            if group.mode == "sequential":
                self.connect(graph, current, node)
                current = node
            else:
                graph.edges.append(
                    Edge(
                        "input",
                        node.id,
                        None,
                        relation="contains",
                        evidence=[self.evidence(path, "structural_membership", "direct")],
                    )
                )
        output = Node("output", "Output", "output", outputs=list(current.outputs))
        graph.nodes.append(output)
        if all(g.mode == "sequential" for g in groups):
            self.connect(graph, current, output)
        else:
            graph.metadata["warning"] = "Execution connectivity is unknown; containment is not dataflow"
        return graph

    def source_graph(self, path: str) -> Graph:
        flow_analyzer = StaticFlowAnalyzer(self.model, path, self.analyzer.classifications)
        flow = flow_analyzer.analyze()
        for container_path in flow.sequential_containers:
            container = self.model.get_submodule(container_path)
            matching_calls = [
                event
                for event in flow.events
                if event.module_path == container_path and "conditional" not in event.metadata
            ]
            if type(container) is torch.nn.ModuleList and matching_calls:
                self.source_sequential_containers[container_path] = {
                    "source_module_path": path,
                    "source_lines": sorted({event.line for event in matching_calls}),
                    "rule": "static loop carries the previous layer output into the next layer input",
                }
            for group in self.analyzer.repetitions.get(container_path, []):
                if group.mode != "parallel":
                    group.mode, group.evidence = (
                        "sequential",
                        "Static source loop carries the previous layer output into the next layer input",
                    )
        graph = Graph(
            self.graph_id(path),
            [],
            [],
            {
                "module_path": path,
                "topology": "static_source_dependencies",
                "runtime_verified": False,
                "warnings": flow.warnings,
                "analysis_scope": {
                    "profile": "canonical_inference",
                    "model_mode": "eval",
                    "cache": "disabled",
                    "training_loss": "excluded",
                    "inputs": "canonical modality inputs",
                },
                "profile_assumptions": flow.profile_assumptions,
                "flow_diagnostics": getattr(flow, "diagnostics", []),
                **(
                    {
                        "shape_basis": "native ViT source and config; fixed resolution, unmasked, no interpolation"
                    }
                    if self.inputs.native_vit
                    else {}
                ),
            },
        )
        self.flow_reports[path] = {
            "warnings": flow.warnings,
            "event_count": len(flow.events),
            "sequential_containers": sorted(flow.sequential_containers),
            "profile_assumptions": flow.profile_assumptions,
            "diagnostics": getattr(flow, "diagnostics", []),
            "analysis_scope": graph.metadata["analysis_scope"],
        }
        if not flow.events:
            graph.metadata["topology"] = "structural_only"
            graph.nodes = [
                self.module_node(p, f"component{i}")
                for i, p in enumerate(self.raw["modules"][path]["children"])
            ]
            graph.metadata["warning"] = (
                "No source dataflow established; component registration order is not execution order"
            )
            return graph
        nodes, used_ids, residual_index = {}, set(), 0

        def unique(name: str) -> str:
            base, index = identifier(name), 1
            candidate = base
            while candidate in used_ids:
                index += 1
                candidate = f"{base}_{index}"
            used_ids.add(candidate)
            return candidate

        used_inputs = {
            ref for event in flow.events for arg in event.arguments.values() for ref, _ in arg.refs
        } | {ref for ref, _ in flow.outputs.refs}
        for name, value in flow.inputs.items():
            ref = value.refs[0][0]
            if ref not in used_inputs:
                continue
            alias = self.analyzer.adapter.input_aliases.get(name, name) if self.analyzer.adapter else name
            spec = self.inputs.spec(name, path)
            node = Node(
                unique(alias),
                name.replace("_", " ").title(),
                "input",
                outputs=[spec],
                metadata={"argument_name": name, "shape_evidence": "symbolic input convention"},
            )
            if name == "input_ids":
                node.label = "Token IDs"
            if name == "position_ids":
                node.label = "Position IDs"
            graph.nodes.append(node)
            nodes[ref] = node
        for event in flow.events:
            if event.module_path is not None:
                last = event.module_path.rsplit(".", 1)[-1]
                alias = self.analyzer.adapter.node_aliases.get(last, last) if self.analyzer.adapter else last
                node = self.call_node(event, unique(alias))
            else:
                embedding_add = "embedding" in self.raw["modules"][path]["type"].lower() or any(
                    "parameter:" in ref
                    or nodes.get(ref)
                    and nodes[ref].kind in {"embedding", "position_embedding"}
                    for arg in event.arguments.values()
                    for ref, _ in arg.refs
                )
                if event.kind == "residual" and embedding_add:
                    node = Node(unique("embedding_add"), "Add embeddings", "add")
                elif event.kind == "residual":
                    residual_index += 1
                    node = Node(unique(f"residual_{residual_index}"), event.label, event.kind)
                else:
                    node = Node(unique(event.kind), event.label, event.kind)
                node.evidence = [self.evidence(path, f"AST {event.kind} at line {event.line}")]
            node.metadata.update(source_line=event.line, **event.metadata)
            port_specs = _output_port_specs(event.metadata)
            contract_required = (
                event.kind in {"split", "tensor_operation"}
                or event.metadata.get("output_contract_required") is True
            )
            if contract_required:
                node.metadata["output_contract_required"] = True
            if port_specs is not None:
                node.metadata["output_port_specs"] = {
                    port: {
                        "shape": spec.shape,
                        "dtype": spec.dtype,
                        "semantic_name": spec.semantic_name,
                    }
                    for port, spec in port_specs.items()
                }
            elif "output_port_specs" in event.metadata:
                # Keep the event visibly unresolved, but prevent a generic primary-port
                # fallback from assigning an input shape to an unvalidated output.
                node.metadata.pop("output_port_specs", None)
                node.metadata["output_port_specs_invalid"] = True
            graph.nodes.append(node)
            event_input_edges = []
            for port, value in event.arguments.items():
                for reference, source_port in value.refs:
                    if reference.startswith("parameter:") and reference not in nodes:
                        parameter_path = reference.removeprefix("parameter:")
                        parameter = self.raw["parameters"].get(parameter_path)
                        if parameter:
                            pnode = Node(
                                unique(parameter_path.rsplit(".", 1)[-1]),
                                parameter_path.rsplit(".", 1)[-1],
                                "position_embedding" if "position" in parameter_path else "parameter",
                                outputs=[
                                    TensorSpec(parameter["shape"], self.inputs.activation_dtype, "parameter")
                                ],
                                parameter_count=parameter["parameter_count"],
                                source_modules=[parameter_path],
                                metadata={"parameter_refs": [parameter_path]},
                                evidence=[Evidence("named_parameters", "direct", parameter_path)],
                            )
                            graph.nodes.append(pnode)
                            nodes[reference] = pnode
                    source = nodes.get(reference)
                    if source:
                        target_port = (
                            "skip"
                            if node.kind == "residual" and port == event.metadata.get("skip_argument")
                            else port
                        )
                        self.connect(
                            graph,
                            source,
                            node,
                            source_port=source_port,
                            target_port=target_port,
                            label="skip" if target_port == "skip" else None,
                        )
                        event_input_edges.append(
                            (port, reference, source_port, graph.edges[-1].tensor, graph.edges[-1])
                        )
            if (
                port_specs is None
                and "output_port_specs" not in event.metadata
                and event.metadata.get("output_contract_required") is True
                and event.module_path is not None
                and callable(getattr(flow_analyzer, "module_input", None))
                and callable(getattr(flow_analyzer, "module_spec", None))
            ):
                primary_input = flow_analyzer.module_input(event.module_path, event.arguments)
                primary_key = next(
                    (name for name, value in event.arguments.items() if value is primary_input), None
                )
                references = [
                    (name, reference)
                    for name, value in event.arguments.items()
                    for reference, _ in value.refs
                ]
                if (
                    primary_key is not None
                    and len(primary_input.refs) == 1
                    and references == [(primary_key, primary_input.refs[0][0])]
                    and len(event_input_edges) == 1
                    and event_input_edges[0][:3]
                    == (primary_key, primary_input.refs[0][0], primary_input.refs[0][1])
                    and event_input_edges[0][3] is not None
                ):
                    reconciled_arguments = dict(event.arguments)
                    reconciled_arguments[primary_key] = replace(
                        primary_input, spec=event_input_edges[0][3]
                    )
                    recovered = flow_analyzer.module_spec(event.module_path, reconciled_arguments)
                    if recovered is not None:
                        port_specs = {"output": recovered}
                        node.metadata["output_port_specs"] = {
                            "output": {
                                "shape": recovered.shape,
                                "dtype": recovered.dtype,
                                "semantic_name": recovered.semantic_name,
                            }
                        }
                        node.metadata["contract"] = "exact_native_torch_primitive.v1"
                        node.metadata["contract_provenance"] = "single_graph_input_edge"
                        incoming_edge = event_input_edges[0][4]
                        node.metadata["contract_input_edge"] = {
                            "source_node": incoming_edge.source,
                            "source_port": incoming_edge.source_port,
                            "target_port": incoming_edge.target_port,
                        }
            incoming = node.inputs[0] if node.inputs else None
            node.outputs = [self.output_spec(node, incoming)]
            if node.kind in {"add", "residual", "gating"} and node.inputs and port_specs is None:
                shape = list(node.inputs[0].shape) if node.inputs[0].shape is not None else None
                for item in node.inputs[1:]:
                    if shape is None or item.shape is None:
                        shape = None
                        break
                    width = max(len(shape), len(item.shape))
                    a, b = [1] * (width - len(shape)) + shape, [1] * (width - len(item.shape)) + item.shape
                    if any(x != y and x != 1 and y != 1 for x, y in zip(a, b)):
                        shape = None
                        break
                    shape = [y if x == 1 else x for x, y in zip(a, b)]
                dtype = node.inputs[0].dtype
                if dtype is None or any(item.dtype != dtype for item in node.inputs[1:]):
                    dtype = None
                node.outputs = [TensorSpec(shape, dtype, "combined_states")]
            if port_specs is not None:
                node.outputs = list(port_specs.values())
            elif ("output_port_specs" in event.metadata or contract_required) and port_specs is None:
                node.outputs = [TensorSpec(None, None, "unresolved_output_contract")]
            nodes[event.id] = node
        output = Node(unique("output"), "Output", "output")
        graph.nodes.append(output)
        for ref, port in flow.outputs.refs:
            if ref in nodes:
                self.connect(graph, nodes[ref], output, source_port=port)
        output.outputs = list(output.inputs) or [TensorSpec(None, None, "unknown")]
        graph.metadata["output_fields"] = {
            name: [[nodes[ref].id, port] for ref, port in value.refs if ref in nodes]
            for name, value in (flow.outputs.fields or {}).items()
        }
        called_paths = {event.module_path for event in flow.events if event.module_path}
        for child_path in self.raw["modules"][path]["children"]:
            if any(p == child_path or p.startswith(child_path + ".") for p in called_paths):
                continue
            classification = self.analyzer.classifications[child_path]
            aliases = [
                group
                for group in self.raw["parameter_alias_groups"]
                if any(p.startswith(child_path + ".") for p in group)
            ]
            if classification.kind != "unknown" and not (classification.kind == "embedding" and aliases):
                continue
            extra = self.module_node(child_path, unique(child_path.rsplit(".", 1)[-1]))
            extra.metadata["invocation_not_established"] = True
            if aliases:
                extra.label = "Shared Embedding"
                for target in list(graph.nodes):
                    if target.module_path and any(
                        alias.startswith(target.module_path + ".") for group in aliases for alias in group
                    ):
                        graph.edges.append(
                            Edge(
                                extra.id,
                                target.id,
                                None,
                                relation="shared_parameter",
                                evidence=[Evidence("parameter_identity", "direct", child_path)],
                            )
                        )
            graph.nodes.append(extra)
        return graph

    def flatten_decoder_backbone(self, graph: Graph) -> None:
        backbone = next(
            (
                n
                for n in graph.nodes
                if n.module_path == self.model.base_model_prefix and n.kind == "decoder" and n.child_graph
            ),
            None,
        )
        if backbone is None:
            return
        child = backbone.child_graph
        incoming = [e for e in graph.edges if e.target == backbone.id]
        outgoing = [e for e in graph.edges if e.source == backbone.id]
        output_ids = {node.id for node in child.nodes if node.kind == "output"}
        output_edges = [edge for edge in child.edges if edge.target in output_ids]
        fields = child.metadata.get("output_fields")
        if not isinstance(fields, Mapping) or not outgoing:
            return
        routes: dict[str, Edge] = {}
        for edge in outgoing:
            references = fields.get(edge.source_port)
            if not isinstance(references, list) or len(references) != 1:
                return
            reference = references[0]
            if not isinstance(reference, list) or len(reference) != 2:
                return
            matches = [
                candidate
                for candidate in output_edges
                if [candidate.source, candidate.source_port] == reference
            ]
            if len(matches) != 1 or matches[0].tensor is None:
                return
            routes[edge.source_port] = matches[0]

        input_routes: dict[str, Edge] = {}
        for node in child.nodes:
            if node.kind == "input":
                argument = node.metadata.get("argument_name")
                matches = [
                    edge
                    for edge in incoming
                    if edge.target_port == argument
                    or (edge.target_port == "0" and argument == "input_ids")
                ]
                if len(matches) > 1:
                    return
                if matches:
                    edge = matches[0]
                    if edge.tensor is None or any(
                        (spec.shape, spec.dtype) != (edge.tensor.shape, edge.tensor.dtype)
                        for spec in node.outputs
                    ):
                        return
                    input_routes[node.id] = edge

        used_ids = {node.id for node in graph.nodes if node is not backbone}
        mapping: dict[str, str] = {}
        inlined_nodes: list[Node] = []
        for node in child.nodes:
            if node.id in output_ids:
                continue
            if node.id in input_routes:
                mapping[node.id] = input_routes[node.id].source
                continue
            new_id = node.id
            if new_id in used_ids:
                new_id = f"{backbone.id}_{new_id}"
                while new_id in used_ids:
                    new_id += "_"
            used_ids.add(new_id)
            mapping[node.id] = new_id
            inlined_nodes.append(
                replace(node, id=new_id, metadata={**node.metadata, "backbone_path": backbone.module_path})
            )

        if any(route.source not in mapping for route in routes.values()):
            return
        inlined_edges: list[Edge] = []
        for edge in child.edges:
            if edge.target in output_ids or edge.source not in mapping or edge.target not in mapping:
                continue
            input_route = input_routes.get(edge.source)
            inlined_edges.append(
                replace(
                    edge,
                    source=mapping[edge.source],
                    target=mapping[edge.target],
                    source_port=input_route.source_port if input_route else edge.source_port,
                    tensor=input_route.tensor if input_route else edge.tensor,
                )
            )
        rewired = []
        for edge in outgoing:
            route = routes[edge.source_port]
            input_route = input_routes.get(route.source)
            rewired.append(
                replace(
                    edge,
                    source=mapping[route.source],
                    source_port=input_route.source_port if input_route else route.source_port,
                    tensor=input_route.tensor if input_route else route.tensor,
                    evidence=[*edge.evidence, *route.evidence],
                )
            )
        graph.nodes = [node for node in graph.nodes if node is not backbone] + inlined_nodes
        graph.edges = [
            edge for edge in graph.edges if edge.source != backbone.id and edge.target != backbone.id
        ] + inlined_edges + rewired
        affected = {edge.target for edge in rewired}
        affected.update(edge.target for edge in inlined_edges)
        for node in graph.nodes:
            if node.id in affected:
                node.inputs = [
                    edge.tensor
                    for edge in graph.edges
                    if edge.target == node.id and edge.relation == "dataflow" and edge.tensor is not None
                ]
        parent_fields = graph.metadata.get("output_fields")
        if isinstance(parent_fields, dict):
            for references in parent_fields.values():
                if isinstance(references, list):
                    for reference in references:
                        if isinstance(reference, list) and len(reference) == 2 and reference[0] == backbone.id:
                            route = routes.get(reference[1])
                            if route is not None:
                                input_route = input_routes.get(route.source)
                                reference[:] = [
                                    mapping[route.source],
                                    input_route.source_port if input_route else route.source_port,
                                ]
        graph.metadata["backbone_path"] = backbone.module_path
        # Render nodes in a stable topological order after inlining the backbone.
        ordered, pending = [], list(graph.nodes)
        while pending:
            ready = [
                node
                for node in pending
                if all(
                    edge.source in {n.id for n in ordered} for edge in graph.edges if edge.target == node.id
                )
            ]
            if not ready:
                ordered.extend(pending)
                break
            for node in ready:
                ordered.append(node)
                pending.remove(node)
        graph.nodes = ordered

    def attention(self, path: str) -> Graph:
        from .attention import build_attention

        return build_attention(self, path)
