"""Generic semantic hierarchy assembled from static source dependencies and patterns."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

import torch

from .inputs import InputResolver
from .ir import Edge, Evidence, Graph, Node, RepetitionSpec, TensorSpec
from .patterns import Repetition, SemanticAnalyzer
from .static_flow import FlowEvent, StaticFlowAnalyzer, StaticValue
from .shape_rules import module_contract
from .shape_values import describe, bind as bind_output, ports as output_ports
from .source_eval import DTYPES

LABELS = {
    "selection_projection": "Select / Project",
    "output_head_group": "Output Heads",
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
        self.call_contexts: dict[str, dict[str, StaticValue]] = {}

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
            "template_equivalence": {"status": self.analyzer.signatures[path].equivalence_status,
                "proof_path": path, "scope": "Computation template, not learned parameter value equality"},
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
            node = self.module_node(groups[0].instance_paths[0], node_id, groups[0], build_child=False)
            node.module_path, node.module_type = path, self.raw["modules"][path]["type"]
            node.metadata["instance_module_type"] = self.raw["modules"][groups[0].instance_paths[0]]["type"]
        else:
            node = self.module_node(path, node_id, build_child=False)
            if groups and node.kind == "container":
                node.kind, node.label = (
                    "block_stack",
                    f"Block Stack ({sum(g.count for g in groups)} instances)",
                )
                node.metadata["variant_groups"] = [dict(asdict(g), count=g.count) for g in groups]
        node.metadata["invocation_established"] = True
        node.child_graph = event.resolved_graph
        if node.child_graph and node.child_graph.metadata.get("attention_proof", {}).get("attention_pattern") == "unknown":
            node.metadata["candidate_kind"] = node.kind
            node.kind, node.label = "unknown", "Unverified attention pattern"
            node.confidence = 0.0
        return node

    def output_spec(self, node: Node, incoming: TensorSpec | None = None) -> TensorSpec:
        path, kind = node.module_path, node.kind
        if "shape_check" in node.metadata or "output_structure" in node.metadata:
            structure = node.metadata.get("output_structure")
            if structure:
                values = list(output_ports(structure).values())
                if values:
                    return TensorSpec(values[0]["shape"], values[0]["dtype"], "source_output")
            output = node.metadata.get("proven_output", {})
            return TensorSpec(output.get("shape"), output.get("dtype"), "checked_output")
        if kind == "source_operation" and node.metadata.get("source_proof"):
            output = node.metadata["proven_output"]
            return TensorSpec(output["shape"], output["dtype"], "source_result")
        if node.metadata.get("unsupported"):
            return TensorSpec(None, None, "unsupported_source_output")
        if node.child_graph:
            structure = node.child_graph.metadata.get("output_structure")
            if structure:
                outputs = list(output_ports(structure).values())
                if outputs:
                    return TensorSpec(outputs[0]["shape"], outputs[0]["dtype"], "child_output")
        return TensorSpec(None, None, "unproven_output")

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
            explicit = source.metadata.get("output_ports", {}).get(source_port)
            primary = source_port in {"output", "0", "last_hidden_state", "logits"}
            known_rotary = source.kind == "rotary_position_embedding" and source_port in {"0", "1"}
            spec = TensorSpec(explicit["shape"], explicit["dtype"], source_port) if explicit is not None else (
                source.outputs[0]
                if "output_ports" not in source.metadata and source.outputs and (primary or known_rotary)
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
        proof = target.metadata.get("source_proof") or source.metadata.get("source_proof")
        if proof:
            graph.edges[-1].evidence.append(Evidence("python_source", "derived", "bounded_python_source",
                                                   {"sources": proof["sources"], "conditions": proof["conditions"],
                                                    "tensor_execution": False}))

    def build(self, path: str = "", arguments: dict[str, StaticValue] | None = None) -> Graph:
        contexts, overrides = dict(self.call_contexts), dict(self.inputs.overrides)
        try:
            return self.build_context(path, arguments)
        finally:
            self.active.discard(path)
            self.call_contexts, self.inputs.overrides = contexts, overrides

    def build_context(self, path: str, arguments: dict[str, StaticValue] | None) -> Graph:
        self.active.add(path)
        if arguments is not None:
            self.call_contexts[path] = arguments
            for name, value in arguments.items():
                self.inputs.overrides[(path, name)] = StaticFlowAnalyzer.tensor_spec(value)
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

    def resolve_module(self, event: FlowEvent, arguments: dict[str, StaticValue]) -> StaticValue:
        path = event.module_path
        module = self.model.get_submodule(path) if path else self.model
        roles = event.metadata.get("argument_roles", {})
        named = {roles.get(name, name): value for name, value in arguments.items()}
        incoming = next((StaticFlowAnalyzer.tensor_spec(value) for value in arguments.values()
                         if value.domain == "tensor" or value.symbolic is not None), None)
        check = module_contract(module, incoming)
        if check is not None:
            structure = {"kind": "tensor", "shape": check.spec.shape, "dtype": check.spec.dtype}
            event.metadata.update(shape_check=check.record(), output_structure=structure,
                                  output_ports=output_ports(structure))
            return bind_output(structure, event.id)
        if path in self.active:
            structure = {"kind": "unknown", "shape": None, "dtype": None}
            event.metadata["shape_check"] = {"rule": "recursive_module", "status": "unknown", "reason": "Recursive module shape contract is not established"}
        else:
            graph = self.build(path, named)
            event.resolved_graph = graph
            structure = graph.metadata.get("output_structure", {"kind": "unknown"})
        event.metadata.update(output_structure=structure, output_ports=output_ports(structure))
        return bind_output(structure, event.id)

    def sequential(self, path: str) -> Graph:
        return self.component_chain(path, None)

    def container(self, path: str) -> Graph:
        return self.component_chain(path, self.analyzer.repetitions[path])

    def component_chain(self, path: str, groups: list[Repetition] | None) -> Graph:
        """Propagate actual caller contracts through sequential representative components."""
        arguments = dict(self.call_contexts.get(path, {}))
        if not arguments:
            spec = self.inputs.spec("hidden_states", path)
            arguments = {"hidden_states": bind_output({"kind": "tensor", "shape": spec.shape,
                                                       "dtype": spec.dtype}, "input")}
        incoming = next(iter(arguments.values()))
        sequential_contract = isinstance(self.model.get_submodule(path), torch.nn.Sequential)
        if groups is None or sequential_contract:
            arguments = {"0": incoming}
        spec = StaticFlowAnalyzer.tensor_spec(incoming)
        graph = Graph(self.graph_id(path), [], [], {"module_path": path,
            "topology": "repeated subtrees" if groups is not None else "nn.Sequential contract",
            "runtime_verified": False})
        current = Node("input", "Input", "input", outputs=[spec])
        graph.nodes.append(current)
        components = [(group.instance_paths[0], group) for group in groups] if groups is not None else [
            (child, None) for child in self.raw["modules"][path]["children"]]
        structure = describe(incoming)
        for index, (child, group) in enumerate(components):
            sequential = group is None or group.mode == "sequential"
            if not sequential:
                node = self.module_node(child, f"group{index}", group, build_child=False)
                node.metadata["invocation_not_established"] = True
                graph.nodes.append(node)
                graph.edges.append(Edge("input", node.id, None, relation="contains"))
                structure = {"kind": "unknown"}
                continue
            roles = StaticFlowAnalyzer(self.model, path, self.analyzer.classifications).module_argument_roles(child, arguments)
            event = FlowEvent(f"component{index}", child, "module", child, arguments, 0, roles)
            value = self.resolve_module(event, arguments)
            node = self.module_node(child, event.id, group, build_child=False)
            node.child_graph = event.resolved_graph
            node.metadata.update(event.metadata, invocation_established=True)
            node.outputs = [TensorSpec(item["shape"], item["dtype"], port)
                            for port, item in node.metadata.get("output_ports", {}).items()]
            for port in current.metadata.get("output_ports", {"output": None}):
                self.connect(graph, current, node, source_port=port, target_port="0")
            graph.nodes.append(node)
            structure = describe(value)
            if group and group.count > 1:
                primary = next(iter(output_ports(structure).values()), {"shape": None, "dtype": None})
                compatible = primary["shape"] == spec.shape and primary["dtype"] == spec.dtype and spec.shape is not None
                node.metadata["repetition_shape_contract"] = {
                    "status": "conditional" if compatible else "unknown",
                    "rule": "representative_shape_invariance", "runtime_verified": False,
                    "reason": "Conditional on existing repeated-template equivalence; E01 remains separate",
                    "input_shape": spec.shape, "output_shape": primary["shape"]}
                if not compatible:
                    structure = {"kind": "unknown"}
            current = node
            if group is None or sequential_contract:
                arguments = {"0": value}
            elif index + 1 < len(components):
                primary_value = value.items[0] if value.items else value
                arguments[next(iter(arguments))] = primary_value
        graph.metadata["output_structure"] = structure
        output = Node("output", "Output", "output", outputs=[TensorSpec(item["shape"], item["dtype"], port)
            for port, item in output_ports(structure).items()], metadata={"output_ports": output_ports(structure)})
        graph.nodes.append(output)
        for port in output_ports(structure):
            self.connect(graph, current, output, source_port=port)
        if groups is not None and any(group.mode != "sequential" for group in groups):
            graph.metadata["warning"] = "Execution connectivity is unknown; containment is not dataflow"
        return graph

    def source_graph(self, path: str) -> Graph:
        if any(child in self.analyzer.repetitions for child in self.raw["modules"][path]["children"]):
            probe = StaticFlowAnalyzer(self.model, path, self.analyzer.classifications,
                                       argument_overrides=self.call_contexts.get(path)).analyze()
            for container_path in probe.sequential_containers:
                for group in self.analyzer.repetitions.get(container_path, []):
                    if group.mode != "parallel":
                        group.mode, group.evidence = "sequential", (
                            "Static source loop carries the previous layer output into the next layer input")
        flow = StaticFlowAnalyzer(self.model, path, self.analyzer.classifications,
                                  self.resolve_module, self.call_contexts.get(path)).analyze()
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
                "diagnostics": flow.diagnostics,
                "forward_wrapper_proof": flow.wrapper_proof,
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
            "resolved_operations": flow.resolved_operations,
            "unknown_inputs": sorted(flow.unknown_inputs),
            "diagnostics": flow.diagnostics,
            "forward_wrapper_proof": flow.wrapper_proof,
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
            spec = StaticFlowAnalyzer.tensor_spec(value) if value.shape_spec is not None or value.symbolic is not None else TensorSpec(None, None, name)
            node = Node(
                unique(alias),
                name.replace("_", " ").title(),
                "input",
                outputs=[spec],
                metadata={"argument_name": name, "shape_evidence":
                          "Unresolved argument; rank and dtype unknown" if name in flow.unknown_inputs
                          else "symbolic input convention"},
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
            if event.metadata.get("source_proof"):
                node.evidence.append(Evidence("python_source", "derived", "bounded_python_source", event.metadata["source_proof"]))
            graph.nodes.append(node)
            for port, value in event.arguments.items():
                for reference, source_port in value.refs:
                    if reference.startswith(("parameter:", "buffer:")) and reference not in nodes:
                        is_buffer = reference.startswith("buffer:")
                        parameter_path = reference.split(":", 1)[1]
                        parameter = self.raw["buffers" if is_buffer else "parameters"].get(parameter_path)
                        if parameter:
                            pnode = Node(
                                unique(parameter_path.rsplit(".", 1)[-1]),
                                parameter_path.rsplit(".", 1)[-1],
                                "buffer" if is_buffer else "parameter",
                                outputs=[
                                    TensorSpec(parameter["shape"], {str(dtype): code for dtype, code in DTYPES.items()}.get(parameter["dtype"], parameter["dtype"]), "buffer" if is_buffer else "parameter")
                                ],
                                parameter_count=0 if is_buffer else parameter["parameter_count"],
                                source_modules=[parameter_path],
                                metadata={"buffer_ref": parameter_path} if is_buffer else {"parameter_refs": [parameter_path]},
                                evidence=[Evidence("named_buffers" if is_buffer else "named_parameters", "direct", parameter_path)],
                            )
                            graph.nodes.append(pnode)
                            nodes[reference] = pnode
                    source = nodes.get(reference)
                    if source:
                        role = event.metadata.get("argument_roles", {}).get(port, port)
                        source_operation = source.metadata.get("unsupported") or source.metadata.get("source_proof")
                        if source_operation:
                            source.metadata["usage_roles"] = sorted(set(source.metadata.get("usage_roles", [])) | {role})
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
                            label="skip" if target_port == "skip" else role if source_operation else None,
                        )
            incoming = node.inputs[0] if node.inputs else None
            node.outputs = [self.output_spec(node, incoming)]
            if node.metadata.get("output_ports"):
                node.outputs = [TensorSpec(item["shape"], item["dtype"], port) for port, item in node.metadata["output_ports"].items()]
            indexing = node.metadata.get("indexing", {})
            input_refs = event.arguments.get("input")
            if (
                indexing.get("shape_proven") and input_refs and len(input_refs.refs) == 1
                and input_refs.refs[0][1] == "output"
                and input_refs.refs[0][0] in nodes
            ):
                flow.resolved_operations.append(dict(indexing, rule="full_slice_identity"))
                graph.nodes.remove(node)
                graph.edges = [edge for edge in graph.edges if edge.target != node.id]
                nodes[event.id] = nodes[input_refs.refs[0][0]]
                continue
            if node.kind in {"add", "residual", "gating"} and node.inputs and "shape_check" not in node.metadata:
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
        graph.metadata["output_structure"] = describe(flow.outputs)
        output.metadata["output_ports"] = output_ports(graph.metadata["output_structure"])
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
        input_ports = {}
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
                    input_ports[node.id] = edge.source_port
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
                edge.source_port = input_ports.get(edge.source, edge.source_port)
                edge.source, edge.target = mapping[edge.source], mapping[edge.target]
                graph.edges.append(edge)
        if primary:
            for edge in outgoing:
                refs = child.metadata.get("output_fields", {}).get(edge.source_port, [])
                resolved = refs[0] if len(refs) == 1 else [primary.source, primary.source_port]
                edge.source = mapping[resolved[0]]
                edge.source_port = resolved[1]
                matching = next((item for item in child.edges if item.target in output_nodes and item.source == resolved[0] and item.source_port == resolved[1]), primary)
                edge.tensor = matching.tensor
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
