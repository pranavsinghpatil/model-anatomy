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
