"""Inspectable semantic classification and structural repetition detection."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from .equivalence import EquivalenceEvidence


@dataclass(frozen=True)
class SemanticClassification:
    kind: str
    confidence: float
    evidence: list[str]


@dataclass(frozen=True)
class StructuralSignature:
    digest: str
    module_count: int
    parameter_count: int
    equivalence_status: str = "unknown"


@dataclass
class Repetition:
    container_path: str
    instance_paths: list[str]
    signature: str
    mode: str = "unknown"
    evidence: str = "Equivalent sibling subtrees; execution mode not established"

    @property
    def count(self) -> int:
        return len(self.instance_paths)


class SemanticAnalyzer:
    def __init__(self, model: torch.nn.Module, raw: dict[str, Any]):
        self.model, self.raw = model, raw
        self.classifications: dict[str, SemanticClassification] = {}
        self.signatures: dict[str, StructuralSignature] = {}
        self.repetitions: dict[str, list[Repetition]] = {}
        collector = EquivalenceEvidence()
        self.equivalence = {path: collector.inspect(model.get_submodule(path) if path else model)
                            for path in raw["modules"]}
        for path in reversed(list(raw["modules"])):
            self.classifications[path] = self.classify(path)
        for path in raw["modules"]:
            self.signature(path)
        for path, record in raw["modules"].items():
            children = record["children"]
            if not children or not (
                record["container"] or all(p.rsplit(".", 1)[-1].isdigit() for p in children)
            ):
                continue
            groups = []
            for child in children:
                digest = self.signatures[child].digest
                if groups and groups[-1].signature == digest:
                    groups[-1].instance_paths.append(child)
                else:
                    mode = "sequential" if record["container"] == "sequential" else "unknown"
                    evidence = (
                        "torch.nn.Sequential guarantees sequential composition"
                        if mode == "sequential"
                        else "ModuleList alone does not establish execution order"
                    )
                    if "expert" in path.lower() and any(
                        self.classifications[p].kind == "router"
                        for p in self.raw["modules"].get(record["parent"], {}).get("children", [])
                    ):
                        mode, evidence = (
                            "parallel",
                            "Expert container plus sibling router topology; routed alternatives, not a sequential stack",
                        )
                    groups.append(Repetition(path, [child], digest, mode, evidence))
            self.repetitions[path] = groups

    def classify(self, path: str) -> SemanticClassification:
        module = self.model.get_submodule(path) if path else self.model
        record = self.raw["modules"][path]
        name, class_name = path.rsplit(".", 1)[-1].lower(), record["type"].lower()
        children = record["children"]
        child_names = {p.rsplit(".", 1)[-1].lower() for p in children}
        descendant_kinds = {c.kind for p, c in self.classifications.items() if p.startswith(path + ".")}
        reasons = [f"class={record['qualified_type']}", f"children={sorted(child_names)}"]

        def result(kind: str, confidence: float, reason: str) -> SemanticClassification:
            return SemanticClassification(kind, confidence, reasons + [reason])

        if not path:
            return result("model", 1.0, "native PreTrainedModel root")
        if name in {"lm_head", "cls", "predictions", "embed_out", "output_projection"} and (
            isinstance(module, torch.nn.Linear) or "mlm" in class_name or "prediction" in class_name
        ):
            return result("lm_head", 0.95, "task-head context and projection topology")
        if "classifier" in name and (isinstance(module, (torch.nn.Linear, torch.nn.Sequential)) or children):
            return result("classifier", 0.96, "classifier ownership context and projection subtree")
        if isinstance(module, torch.nn.Embedding):
            return result(
                "position_embedding" if "position" in name or name == "wpe" else "embedding",
                1.0,
                "isinstance(nn.Embedding), with positional ownership context",
            )
        if isinstance(module, torch.nn.LayerNorm):
            return result("layer_norm", 1.0, "isinstance(nn.LayerNorm)")
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return result("batch_norm", 1.0, "isinstance(nn._BatchNorm)")
        if (
            "norm" in class_name
            and len(record["parameters"]) <= 2
            and any(len(self.raw["parameters"][p]["shape"]) == 1 for p in record["parameters"])
        ):
            return result(
                "rms_norm"
                if "rms" in class_name or not any(p.endswith("bias") for p in record["parameters"])
                else "normalization",
                0.9,
                "normalization class plus one-dimensional affine parameter topology",
            )
        if isinstance(module, torch.nn.Dropout):
            return result("dropout", 1.0, "isinstance(nn.Dropout)")
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            return result("convolution", 1.0, "isinstance(nn._ConvNd), kernel and stride metadata")
        if isinstance(module, torch.nn.Linear) or (
            record["type"] == "Conv1D" and "nf" in record["attributes"]
        ):
            parent_children = self.raw["modules"].get(record["parent"], {}).get("children", [])
            routed = any("expert" in p.rsplit(".", 1)[-1].lower() for p in parent_children)
            return result(
                "router" if name in {"router", "gate"} and routed else "projection",
                0.98,
                "affine projection type and parameter dimensions",
            )
        if "pool" in class_name and (not children or "pooling" in type(module).__module__):
            return result("pooling", 0.98, "pooling implementation and ownership context")
        if isinstance(module, torch.nn.Identity):
            return result("identity", 1.0, "isinstance(nn.Identity)")
        if isinstance(module, torch.nn.Flatten):
            return result("reshape", 1.0, "isinstance(nn.Flatten)")
        if (
            "activation" in type(module).__module__
            or "activations" in type(module).__module__
            or isinstance(
                module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Tanh, torch.nn.Sigmoid)
            )
        ):
            return result("activation", 0.99, "activation implementation namespace and primitive type")
        if {"gate_proj", "up_proj", "down_proj"}.issubset(child_names) or {"wi_0", "wi_1", "wo"}.issubset(
            child_names
        ):
            return result("gated_mlp", 0.99, "gate/expansion/contraction projection topology")
        qkv = any(
            names.issubset(child_names)
            for names in ({"q_proj", "k_proj", "v_proj"}, {"query", "key", "value"}, {"q", "k", "v"})
        )
        fused = "c_attn" in child_names and "c_proj" in child_names
        if qkv or fused:
            cross = record["attributes"].get("is_cross_attention") or "cross" in name or "encdec" in name
            return result(
                "cross_attention" if cross else "self_attention",
                0.99,
                "Q/K/V or fused-QKV parameter topology, with attention role context",
            )
        if any(names.issubset(child_names) for names in ({"c_fc", "c_proj"}, {"fc1", "fc2"}, {"wi", "wo"})):
            return result("feed_forward", 0.97, "expansion/contraction projections with activation subtree")
        if "projection" in descendant_kinds and "activation" in descendant_kinds and "intermediate" in name:
            return result(
                "feed_forward",
                0.85,
                "intermediate expansion projection plus activation, within a feed-forward stage",
            )
        attention = bool(descendant_kinds & {"self_attention", "cross_attention", "attention"})
        feed_forward = bool(descendant_kinds & {"feed_forward", "gated_mlp"})
        if (
            attention
            and feed_forward
            and ("block" in class_name or "layer" in class_name)
            and not record["container"]
        ):
            decoder = (
                record["attributes"].get("is_decoder") or "decoder" in path.lower() or "decoder" in class_name
            )
            if not decoder and "encoder" not in path.lower():
                decoder = any(
                    "causallm" in a.lower() or "lmheadmodel" in a.lower()
                    for a in (self.model.config.architectures or [])
                )
            return result(
                "decoder_block" if decoder else "encoder_block",
                0.96,
                "normalization + attention + feed-forward subtree, with encoder/decoder context",
            )
        if "patch" in class_name and "convolution" in descendant_kinds:
            return result("patch_embedding", 0.98, "patch ownership plus convolutional projection")
        if "embedding" in class_name and (
            "embedding" in descendant_kinds or "patch_embedding" in descendant_kinds
        ):
            return result("embeddings", 0.97, "composite embedding subtree")
        if "rotary" in class_name and hasattr(module, "inv_freq"):
            return result("rotary_position_embedding", 0.98, "rotary class and frequency buffer topology")
        if "convolution" in descendant_kinds and "batch_norm" in descendant_kinds and "layer" in class_name:
            try:
                source = inspect.getsource(type(module).forward)
            except (OSError, TypeError):
                source = ""
            if "residual" in source and ("+=" in source or " + " in source):
                return result(
                    "residual_block",
                    0.97,
                    "convolution/normalization subtree and explicit residual addition in source",
                )
        if (
            children
            and "convolution" in descendant_kinds
            and "batch_norm" in descendant_kinds
            and not attention
            and all(
                self.classifications[p].kind in {"convolution", "batch_norm", "activation", "identity"}
                for p in children
            )
        ):
            return result(
                "convolution_block",
                0.98,
                "direct convolution + normalization + optional activation composition",
            )
        if (
            children
            and "projection" in descendant_kinds
            and "layer_norm" in descendant_kinds
            and not attention
            and all(
                self.classifications[p].kind in {"projection", "layer_norm", "dropout", "activation"}
                for p in children
            )
        ):
            return result(
                "projection_block", 0.94, "direct affine projection + normalization/activation composition"
            )
        if (
            type(module).__module__ == "transformers.models.vit.modeling_vit"
            and record["type"] in {"ViTSelfOutput", "ViTOutput"}
            and child_names == {"dense", "dropout"}
            and isinstance(module.dense, torch.nn.Linear)
            and isinstance(module.dropout, torch.nn.Dropout)
        ):
            return result(
                "projection_block",
                0.99,
                "native ViT output projection/dropout; residual connectivity comes from forward source",
            )
        if attention and "attention" in class_name:
            return result(
                "cross_attention" if "cross" in class_name else "self_attention",
                0.91,
                "attention wrapper containing classified Q/K/V subtree",
            )
        if feed_forward and any(token in class_name for token in ("ff", "mlp", "feedforward")):
            return result(
                "feed_forward", 0.92, "feed-forward wrapper containing classified projection pattern"
            )
        if "encoder" in name or "encoder" in class_name:
            return result("encoder", 0.88, "encoder context and contained computation")
        if "decoder" in name or record["attributes"].get("is_decoder"):
            return result("decoder", 0.9, "decoder context or native is_decoder attribute")
        if "model" in class_name and any(
            self.classifications.get(p, SemanticClassification("unknown", 0, [])).kind
            in {"decoder_block", "transformer_block"}
            for p in self.raw["modules"]
            if p.startswith(path + ".")
        ):
            return result("decoder", 0.94, "embedding + repeated decoder-block backbone")
        if "expert" in name or "expert" in class_name:
            return result(
                "expert" if not record["container"] else "expert_group",
                0.8,
                "expert ownership context; routing remains separately reported",
            )
        if "stage" in class_name and "convolution" in descendant_kinds:
            return result("residual_stage", 0.93, "stage boundary containing convolutional residual blocks")
        if "stem" in name or ("embedder" in name and "convolution" in descendant_kinds):
            return result("stem", 0.9, "input stem context with convolutional subtree")
        if record["container"]:
            return result("container", 1.0, "native module container; no execution topology assumed")
        if path == self.raw["base_model_prefix"] and children:
            return result("backbone", 0.99, "native base_model_prefix identifies backbone ownership")
        return result("unknown", 0.0, "No sufficiently supported semantic rule; preserve raw module facts")

    def signature(self, path: str) -> StructuralSignature:
        if path in self.signatures:
            return self.signatures[path]
        entries, unique_parameters = [], {}
        established = True
        prefix = path + "." if path else ""
        for p, module in self.raw["modules"].items():
            if p != path and not p.startswith(prefix):
                continue
            proof = self.equivalence[p]
            established = established and proof["status"] == "established"
            parameters = [
                (name.rsplit(".", 1)[-1], item["shape"], item["dtype"] or item["module_dtype"])
                for name in module["parameters"]
                if (item := self.raw["parameters"].get(name)) is not None
            ]
            for name in module["parameters"]:
                item = self.raw["parameters"][name]
                unique_parameters[item["canonical_parameter"]] = item["parameter_count"]
            buffers = [
                (
                    name.rsplit(".", 1)[-1],
                    self.raw["buffers"][name]["shape"],
                    self.raw["buffers"][name]["dtype"],
                )
                for name in module["buffers"]
            ]
            sharing = [
                [
                    alias[len(prefix) :]
                    for alias in self.raw["parameters"][name]["aliases"]
                    if alias.startswith(prefix)
                ]
                for name in module["parameters"]
            ]
            entries.append(
                [
                    p[len(path) :],
                    module["qualified_type"],
                    module["attributes"],
                    parameters,
                    buffers,
                    sharing,
                    self.classifications[p].kind,
                    self.raw["source_fingerprints"][p]["sha256"],
                    proof,
                ]
            )
        # Independent unproven states must never compare equal by omission.
        digest = hashlib.sha256(json.dumps([entries, None if established else path], sort_keys=True).encode()).hexdigest()
        result = StructuralSignature(digest, len(entries), sum(unique_parameters.values()),
                                     "established" if established else "unknown")
        self.signatures[path] = result
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {
            "classifications": {p: asdict(c) for p, c in self.classifications.items()},
            "structural_signatures": {p: asdict(s) for p, s in self.signatures.items()},
            "template_equivalence": self.equivalence,
            "repetitions": {
                p: [dict(asdict(r), count=r.count) for r in groups] for p, groups in self.repetitions.items()
            },
            "unknown_components": [p for p, c in self.classifications.items() if c.kind == "unknown"],
        }
