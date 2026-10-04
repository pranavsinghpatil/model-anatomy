"""GPT-2-specific semantic rules grounded in config, modules, and source."""

from __future__ import annotations

from typing import Any

from .errors import ArtifactError
from .inspection import GPT2Dimensions, detect_repetition
from .ir import Edge, Evidence, Graph, ModelIR, Node, TensorSpec


class GPT2GraphBuilder:
    def __init__(self, dimensions: GPT2Dimensions, raw: dict[str, Any]):
        self.d = dimensions
        self.raw = raw
        self.paths = detect_repetition(raw, dimensions)
        self.hidden = self.tensor(["B", "S", dimensions.hidden], "hidden_states")

    @staticmethod
    def tensor(shape: list[int | str], name: str, dtype: str = "F32") -> TensorSpec:
        return TensorSpec(shape, dtype, name)

    def rule(self, source_path: str, rule: str) -> Evidence:
        return Evidence(
            "gpt2_semantic_rules",
            "reconstructed",
            rule,
            {
                "module_path": source_path,
                "source_fingerprint": self.raw["source_fingerprints"][source_path],
                "runtime_verified": False,
            },
        )

    def node(
        self,
        id: str,
        label: str,
        kind: str,
        inputs: list[TensorSpec],
        outputs: list[TensorSpec],
        path: str | None = None,
        child: Graph | None = None,
        source: str = "transformer.h.0",
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        details = dict(metadata or {})
        evidence = [self.rule(source, f"{kind}:{id}")]
        module_type = None
        count = 0
        if path is not None:
            module = self.raw["modules"].get(path)
            if module is None:
                raise ArtifactError(f"Missing module required by GPT-2 rules: {path}")
            module_type = module["type"]
            evidence.append(Evidence("named_modules", "direct", path))
            records = [v for p, v in self.raw["parameters"].items() if p.startswith(path + ".")]
            unique = {v["canonical_parameter"]: v for v in records}
            count = sum(v["parameter_count"] for v in unique.values())
            details["parameter_refs"] = [v["name"] for v in records]
            details["parameter_shapes"] = {v["name"]: v["shape"] for v in records}
            details["parameter_count_scope"] = (
                "unique parameters referenced by this component, including shared aliases"
            )
            if records:
                evidence.append(
                    Evidence(
                        "safetensors+named_parameters",
                        "reconstructed",
                        path,
                        {"tensor_names": sorted({v["tensor_name"] for v in records})},
                    )
                )
            for attr in ("dropout_probability", "epsilon"):
                if attr in module:
                    details[attr] = module[attr]
        return Node(
            id,
            label,
            kind,
            path,
            module_type,
            inputs=inputs,
            outputs=outputs,
            parameter_count=count,
            metadata=details,
            evidence=evidence,
            child_graph=child,
        )

    def edge(
        self,
        source: str,
        target: str,
        tensor: TensorSpec | None = None,
        source_port: str = "output",
        target_port: str = "input",
        label: str | None = None,
        rule_source: str = "transformer.h.0",
    ) -> Edge:
        return Edge(
            source,
            target,
            tensor or self.hidden,
            source_port,
            target_port,
            label,
            [self.rule(rule_source, f"dataflow:{source}->{target}")],
        )

    def graph(self, id: str, nodes: list[Node], edges: list[Edge]) -> Graph:
        return Graph(
            id,
            nodes,
            edges,
            {
                "flow_scope": "eval mode, full sequence, no KV cache, no external attention mask",
                "shape_evidence": "symbolically derived from config and GPT-2 semantic rules",
                "runtime_verified": False,
            },
        )

    def attention_core(self) -> Graph:
        d = self.d
        packed = self.tensor(["B", "S", 3 * d.hidden], "packed_qkv")
        heads = self.tensor(["B", d.heads, "S", d.head_dimension], "head_states")
        scores = self.tensor(["B", d.heads, "S", "S"], "attention_scores")
        context = self.tensor(["B", "S", d.heads, d.head_dimension], "transposed_context")
        source = "transformer.h.0.attn"

        def n(
            id: str,
            label: str,
            kind: str,
            inputs: list[TensorSpec],
            outputs: list[TensorSpec],
            path: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> Node:
            return self.node(id, label, kind, inputs, outputs, path, source=source, metadata=metadata)

        nodes = [
            n("input", "Packed QKV", "input", [packed], [packed]),
            n(
                "split",
                "Split QKV and reshape heads",
                "split_heads",
                [packed],
                [heads, heads, heads],
                metadata={"split_size": d.hidden, "heads": d.heads, "head_dimension": d.head_dimension},
            ),
            n("q", "Q", "query", [heads], [heads]),
            n("k", "K", "key", [heads], [heads]),
            n("v", "V", "value", [heads], [heads]),
            n(
                "scores",
                "Q @ K transpose",
                "matmul",
                [heads, heads],
                [scores],
                metadata={"transpose_key_axes": [-1, -2]},
            ),
            n(
                "scale",
                "Scale attention scores",
                "scale",
                [scores],
                [scores],
                metadata={"factor": 1 / d.head_dimension**0.5, "formula": "1 / sqrt(head_dimension)"},
            ),
            n(
                "mask",
                "Causal mask",
                "causal_mask",
                [scores],
                [scores],
                metadata={
                    "rule": "mask key positions greater than query position",
                    "mask_shape": [1, 1, "S", "S"],
                },
            ),
            n("softmax", "Softmax", "softmax", [scores], [scores], metadata={"axis": -1}),
            n(
                "dropout",
                "Attention dropout",
                "dropout",
                [scores],
                [scores],
                "transformer.h.0.attn.attn_dropout",
            ),
            n("aggregate", "Attention probabilities @ V", "matmul", [scores, heads], [heads]),
            n("transpose", "Transpose sequence and head axes", "transpose", [heads], [context]),
            n("merge", "Merge heads", "merge_heads", [context], [self.hidden]),
            n("output", "Context", "output", [self.hidden], [self.hidden]),
        ]
        links = [
            ("input", "split", packed, "output", "qkv"),
            ("split", "q", heads, "q", "input"),
            ("split", "k", heads, "k", "input"),
            ("split", "v", heads, "v", "input"),
            ("q", "scores", heads, "output", "q"),
            ("k", "scores", heads, "output", "k"),
            ("scores", "scale", scores, "output", "input"),
            ("scale", "mask", scores, "output", "input"),
            ("mask", "softmax", scores, "output", "input"),
            ("softmax", "dropout", scores, "output", "input"),
            ("dropout", "aggregate", scores, "output", "probabilities"),
            ("v", "aggregate", heads, "output", "v"),
            ("aggregate", "transpose", heads, "output", "input"),
            ("transpose", "merge", context, "output", "input"),
            ("merge", "output", self.hidden, "output", "input"),
        ]
        return self.graph(
            "attention_core",
            nodes,
            [self.edge(a, b, t, sp, tp, rule_source=source) for a, b, t, sp, tp in links],
        )

    def attention(self) -> Graph:
        packed = self.tensor(["B", "S", self.d.hidden * 3], "packed_qkv")
        source = "transformer.h.0.attn"
        nodes = [
            self.node("input", "Input", "input", [self.hidden], [self.hidden], source=source),
            self.node(
                "qkv",
                "QKV Projection",
                "projection",
                [self.hidden],
                [packed],
                source + ".c_attn",
                source=source,
                metadata={
                    "in_features": self.d.hidden,
                    "out_features": self.d.hidden * 3,
                    "weight_layout": "input_features, output_features",
                },
            ),
            self.node(
                "core",
                "Attention Core",
                "attention_core",
                [packed],
                [self.hidden],
                child=self.attention_core(),
                source=source,
            ),
            self.node(
                "projection",
                "Output Projection",
                "projection",
                [self.hidden],
                [self.hidden],
                source + ".c_proj",
                source=source,
                metadata={"in_features": self.d.hidden, "out_features": self.d.hidden},
            ),
            self.node(
                "dropout",
                "Residual dropout",
                "dropout",
                [self.hidden],
                [self.hidden],
                source + ".resid_dropout",
                source=source,
            ),
            self.node("output", "Output", "output", [self.hidden], [self.hidden], source=source),
        ]
        pairs = [
            ("input", "qkv", self.hidden),
            ("qkv", "core", packed),
            ("core", "projection", self.hidden),
            ("projection", "dropout", self.hidden),
            ("dropout", "output", self.hidden),
        ]
        return self.graph(
            "attention", nodes, [self.edge(a, b, tensor, rule_source=source) for a, b, tensor in pairs]
        )

    def feed_forward(self) -> Graph:
        inner = self.tensor(["B", "S", self.d.inner], "feed_forward_states")
        source = "transformer.h.0.mlp"
        nodes = [
            self.node("input", "Input", "input", [self.hidden], [self.hidden], source=source),
            self.node(
                "expand",
                "Expand projection",
                "projection",
                [self.hidden],
                [inner],
                source + ".c_fc",
                source=source,
            ),
            self.node(
                "activation",
                self.d.activation,
                "activation",
                [inner],
                [inner],
                source + ".act",
                source=source,
            ),
            self.node(
                "project",
                "Output projection",
                "projection",
                [inner],
                [self.hidden],
                source + ".c_proj",
                source=source,
            ),
            self.node(
                "dropout",
                "Dropout",
                "dropout",
                [self.hidden],
                [self.hidden],
                source + ".dropout",
                source=source,
            ),
            self.node("output", "Output", "output", [self.hidden], [self.hidden], source=source),
        ]
        pairs = [
            ("input", "expand", self.hidden),
            ("expand", "activation", inner),
            ("activation", "project", inner),
            ("project", "dropout", self.hidden),
            ("dropout", "output", self.hidden),
        ]
        return self.graph(
            "feed_forward", nodes, [self.edge(a, b, tensor, rule_source=source) for a, b, tensor in pairs]
        )

    def block(self) -> Graph:
        nodes = [
            self.node("input", "Input", "input", [self.hidden], [self.hidden]),
            self.node(
                "ln_1", "LayerNorm", "normalization", [self.hidden], [self.hidden], "transformer.h.0.ln_1"
            ),
            self.node(
                "attention",
                "Multi-Head Attention",
                "attention",
                [self.hidden],
                [self.hidden],
                "transformer.h.0.attn",
                self.attention(),
            ),
            self.node("residual_1", "Residual Add", "residual", [self.hidden, self.hidden], [self.hidden]),
            self.node(
                "ln_2", "LayerNorm", "normalization", [self.hidden], [self.hidden], "transformer.h.0.ln_2"
            ),
            self.node(
                "mlp",
                "Feed Forward / MLP",
                "feed_forward",
                [self.hidden],
                [self.hidden],
                "transformer.h.0.mlp",
                self.feed_forward(),
            ),
            self.node("residual_2", "Residual Add", "residual", [self.hidden, self.hidden], [self.hidden]),
            self.node("output", "Output", "output", [self.hidden], [self.hidden]),
        ]
        pairs = [
            ("input", "ln_1", "input"),
            ("ln_1", "attention", "input"),
            ("attention", "residual_1", "branch"),
            ("input", "residual_1", "skip"),
            ("residual_1", "ln_2", "input"),
            ("ln_2", "mlp", "input"),
            ("mlp", "residual_2", "branch"),
            ("residual_1", "residual_2", "skip"),
            ("residual_2", "output", "input"),
        ]
        graph = self.graph(
            "transformer_block",
            nodes,
            [
                self.edge(a, b, target_port=port, label="skip" if port == "skip" else None)
                for a, b, port in pairs
            ],
        )
        graph.metadata.update(
            {
                "template_module_path": self.paths[0],
                "instance_paths": self.paths,
                "parameter_refs_scope": "template references block 0; all instance records remain in raw_structure",
            }
        )
        return graph

    def overview(self) -> Graph:
        ids = self.tensor(["B", "S"], "token_ids", "I64")
        positions = self.tensor([1, "S"], "position_ids", "I64")
        position_embedding = self.tensor([1, "S", self.d.hidden], "position_embedding")
        logits = self.tensor(["B", "S", self.d.vocabulary], "logits")
        nodes = [
            self.node("tokens", "Token IDs", "input", [ids], [ids], source="transformer"),
            self.node("positions", "Position IDs", "input", [positions], [positions], source="transformer"),
            self.node(
                "wte",
                "Token Embedding",
                "embedding",
                [ids],
                [self.hidden],
                "transformer.wte",
                source="transformer",
            ),
            self.node(
                "wpe",
                "Position Embedding",
                "embedding",
                [positions],
                [position_embedding],
                "transformer.wpe",
                source="transformer",
            ),
            self.node(
                "embedding_add",
                "Add embeddings",
                "add",
                [self.hidden, position_embedding],
                [self.hidden],
                source="transformer",
                metadata={"broadcast_position_embedding_over_batch": True},
            ),
            self.node(
                "dropout",
                "Dropout",
                "dropout",
                [self.hidden],
                [self.hidden],
                "transformer.drop",
                source="transformer",
            ),
            self.node(
                "blocks",
                "Transformer Block",
                "transformer_block",
                [self.hidden],
                [self.hidden],
                "transformer.h",
                self.block(),
                source="transformer",
            ),
            self.node(
                "ln_f",
                "Final LayerNorm",
                "normalization",
                [self.hidden],
                [self.hidden],
                "transformer.ln_f",
                source="transformer",
            ),
            self.node("lm_head", "LM Head", "projection", [self.hidden], [logits], "lm_head", source=""),
            self.node("output", "Logits", "output", [logits], [logits], source=""),
        ]
        block = next(n for n in nodes if n.id == "blocks")
        block.repeat = len(self.paths)
        block.metadata.update(
            {
                "instance_paths": self.paths,
                "template_path": self.paths[0],
                "per_instance_parameter_count": block.parameter_count // len(self.paths),
                "repeat_execution": "sequential; output of block i feeds block i+1",
                "equivalence": "module types, topology, parameter shapes/dtypes, dropout and normalization attributes",
                "instance_module_type": "GPT2Block",
            }
        )
        block.evidence.append(
            Evidence(
                "structural_equivalence+config",
                "derived",
                "transformer.h",
                {"discovered_instances": len(self.paths), "configured_layers": self.d.layers},
            )
        )
        head = next(n for n in nodes if n.id == "lm_head")
        head.metadata.update(
            {"shared_with": "transformer.wte.weight", "additional_unique_parameter_count": 0}
        )
        pairs = [
            ("tokens", "wte", ids),
            ("positions", "wpe", positions),
            ("wte", "embedding_add", self.hidden),
            ("wpe", "embedding_add", position_embedding),
            ("embedding_add", "dropout", self.hidden),
            ("dropout", "blocks", self.hidden),
            ("blocks", "ln_f", self.hidden),
            ("ln_f", "lm_head", self.hidden),
            ("lm_head", "output", logits),
        ]
        graph = self.graph(
            "overview",
            nodes,
            [
                self.edge(a, b, tensor, rule_source="" if a == "lm_head" or b == "lm_head" else "transformer")
                for a, b, tensor in pairs
            ],
        )
        graph.validate()
        return graph


def build_semantic_ir(
    model_id: str,
    dimensions: GPT2Dimensions,
    raw: dict[str, Any],
    metadata: dict[str, Any],
    tensors: dict[str, dict[str, Any]],
    evidence: list[Evidence],
) -> ModelIR:
    builder = GPT2GraphBuilder(dimensions, raw)
    return ModelIR(model_id, "GPT2LMHeadModel", builder.overview(), metadata, tensors, raw, evidence)
