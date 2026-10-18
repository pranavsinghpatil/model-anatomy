"""Generic semantic hierarchy assembled from static source dependencies and patterns."""

from __future__ import annotations

import re
from dataclasses import asdict
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


def identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value) or "model"


class HierarchyBuilder:
    def __init__(self, model: torch.nn.Module, raw: dict[str, Any], analyzer: SemanticAnalyzer):
        self.model, self.raw, self.analyzer = model, raw, analyzer
        self.inputs = InputResolver(model)
        self.graph_ids: set[str] = set()
        self.flow_reports: dict[str, dict[str, Any]] = {}
        self.active: set[str] = set()

    def graph_id(self, path: str, suggested: str | None = None) -> str:
        if not path:
            suggested = "overview"
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
        node.metadata["invocation_established"] = True
        return node

    def output_spec(self, node: Node, incoming: TensorSpec | None = None) -> TensorSpec:
        path, kind = node.module_path, node.kind
        if node.metadata.get("unsupported"):
            return TensorSpec(None, None, "unsupported_source_output")
        default = self.inputs.spec("hidden_states", path or "")
        if path is None:
            if self.inputs.native_vit and kind in {"reshape", "selection", "concatenation"}:
                return operation_shape(kind, node.metadata, node.inputs)
            if kind in {"matmul", "concatenation", "operation"}:
                return TensorSpec(None, None, "unresolved_operation_output")
            return incoming or TensorSpec(None, None, "result")
        module = self.model.get_submodule(path)
        if self.inputs.native_vit and isinstance(module, torch.nn.Linear):
            shape = list(incoming.shape) if incoming and incoming.shape else None
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
            shape = (
                list(incoming.shape)
                if incoming and incoming.shape
                else list(default.shape)
                if default.shape
                else None
            )
            if shape:
                shape[-1] = module.out_features
            return TensorSpec(shape, self.inputs.activation_dtype, "projected_states")
        if self.raw["modules"][path]["type"] == "Conv1D":
            shape = (
                list(incoming.shape or default.shape)
                if incoming and (incoming.shape or default.shape)
                else default.shape
            )
            shape = list(shape) if shape else None
            if shape:
                shape[-1] = module.nf
            return TensorSpec(shape, self.inputs.activation_dtype, "projected_states")
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
            return incoming or default
        if node.child_graph:
            output_node = next((n for n in node.child_graph.nodes if n.kind == "output"), None)
            if output_node and output_node.outputs:
                return output_node.outputs[0]
        return default if kind != "unknown" else TensorSpec(None, None, "unknown")

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
            primary = source_port in {"output", "0", "last_hidden_state", "logits"}
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
            node.metadata["invocation_established"] = True
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
                node.metadata["invocation_established"] = True
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
        flow = StaticFlowAnalyzer(self.model, path, self.analyzer.classifications).analyze()
        for container_path in flow.sequential_containers:
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
                "analysis_scope": "eval, no cache, no training loss, canonical modality inputs",
                "profile_assumptions": flow.profile_assumptions,
                "input_profile": flow.input_profile,
                "branch_conditions": flow.branch_conditions,
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
            "input_profile": flow.input_profile,
            "branch_conditions": flow.branch_conditions,
            "unexplained_operations": flow.unexplained_operations,
            "call_sites": flow.call_sites,
            "source_call_sites": flow.source_call_sites,
        }
        if not flow.events:
            graph.metadata["topology"] = "structural_only"
            graph.nodes = [
                self.module_node(p, f"component{i}")
                for i, p in enumerate(self.raw["modules"][path]["children"])
            ]
            for node in graph.nodes:
                node.metadata["invocation_not_established"] = True
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
            alias = name
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
                alias = last
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
            graph.nodes.append(node)
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
            incoming = node.inputs[0] if node.inputs else None
            node.outputs = [self.output_spec(node, incoming)]
            if node.kind in {"add", "residual", "gating"} and node.inputs:
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
                node.outputs = [TensorSpec(shape, node.inputs[0].dtype, "combined_states")]
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
            aliases = [
                group
                for group in self.raw["parameter_alias_groups"]
                if any(p.startswith(child_path + ".") for p in group)
            ]
            excluded = [call for call in flow.source_call_sites if call["module_path"] == child_path]
            profile_excluded = bool(excluded) and all(call["excluded_by_profile"] for call in excluded)
            extra = self.module_node(child_path, unique(child_path.rsplit(".", 1)[-1]), build_child=False)
            extra.metadata["invocation_not_established"] = True
            extra.metadata["excluded_by_profile"] = profile_excluded
            if profile_excluded:
                extra.metadata["profile_conditions"] = [call["profile_condition"] for call in excluded]
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
        mapping = {}
        for node in child.nodes:
            if node.kind == "input":
                argument = node.metadata.get("argument_name")
                edge = next(
                    (
                        e
                        for e in incoming
                        if e.target_port == argument or e.target_port == "0" and argument == "input_ids"
                    ),
                    None,
                )
                if edge:
                    mapping[node.id] = edge.source
                    continue
            if node.kind == "output":
                continue
            new_id = node.id if all(n.id != node.id for n in graph.nodes) else backbone.id + "_" + node.id
            mapping[node.id] = new_id
            node.id = new_id
            node.metadata["backbone_path"] = backbone.module_path
            graph.nodes.append(node)
        output_nodes = {n.id for n in child.nodes if n.kind == "output"}
        primary = next((e for e in reversed(child.edges) if e.target in output_nodes), None)
        graph.nodes = [n for n in graph.nodes if n is not backbone]
        graph.edges = [e for e in graph.edges if e.source != backbone.id and e.target != backbone.id]
        for edge in child.edges:
            if edge.target not in output_nodes and edge.source in mapping and edge.target in mapping:
                edge.source, edge.target = mapping[edge.source], mapping[edge.target]
                graph.edges.append(edge)
        if primary:
            for edge in outgoing:
                edge.source = mapping[primary.source]
                edge.tensor = primary.tensor
                graph.edges.append(edge)
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
