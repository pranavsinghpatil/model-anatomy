"""Module roles from native identities and registered parameter/child topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SemanticClassification:
    kind: str
    confidence: float
    evidence: list[str]


class SemanticAnalyzer:
    """Annotate actual modules without claiming computational equivalence."""

    def __init__(self, model: torch.nn.Module, raw: dict[str, Any]):
        self.model, self.raw = model, raw
        self.classifications: dict[str, SemanticClassification] = {}
        for path in reversed(list(raw["modules"])):
            self.classifications[path] = self.classify(path)

    def classify(self, path: str) -> SemanticClassification:
        module = self.model.get_submodule(path) if path else self.model
        record = self.raw["modules"][path]
        name, class_name = path.rsplit(".", 1)[-1].lower(), record["type"].lower()
        children = record["children"]
        native = type(module).__module__.startswith("transformers.")
        reasons = [f"class={record['qualified_type']}", f"registered_children={children}"]

        def result(kind: str, reason: str, confidence: float = 1.0) -> SemanticClassification:
            return SemanticClassification(kind, confidence, [*reasons, reason])

        if not path:
            return result("model", "Native constructed model root")
        if isinstance(module, torch.nn.Embedding):
            return result("position_embedding" if "position" in name or name == "wpe" else "embedding",
                          "nn.Embedding identity; positional role uses ownership context")
        if isinstance(module, torch.nn.LayerNorm):
            return result("layer_norm", "nn.LayerNorm identity")
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return result("batch_norm", "nn._BatchNorm identity")
        if isinstance(module, torch.nn.RMSNorm):
            return result("rms_norm", "nn.RMSNorm identity")
        if native and "norm" in class_name and not children and record["parameters"] and all(
            len(self.raw["parameters"][p]["shape"]) == 1 for p in record["parameters"]
        ):
            return result("rms_norm" if "rms" in class_name else "normalization",
                          "Native normalization class with one-dimensional affine state", 0.95)
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            return result("dropout", "Native Dropout module identity")
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            return result("convolution", "Native convolution type and kernel metadata")
        if isinstance(module, torch.nn.Linear) or (
            native and record["type"] == "Conv1D" and "nf" in record["attributes"]
        ):
            kind = ("classifier" if name in {"classifier", "classification_head", "score"} else
                    "lm_head" if name in {"lm_head", "embed_out", "output_projection", "proj_out"} else "projection")
            return result(kind, "Affine module identity and parameter dimensions; task role uses ownership context")
        if isinstance(module, torch.nn.Identity):
            return result("identity", "nn.Identity identity")
        if isinstance(module, torch.nn.Flatten):
            return result("reshape", "nn.Flatten identity")
        namespace = type(module).__module__
        if namespace in {"torch.nn.modules.activation", "transformers.activations"}:
            return result("activation", "Native activation implementation identity")
        if namespace == "torch.nn.modules.pooling":
            return result("pooling", "Native pooling implementation identity")
        if record["container"]:
            return result("container", "Registered native container; no execution order implied")

        descendants = {p: c.kind for p, c in self.classifications.items()
                       if p.startswith(path + ".")}
        kinds = set(descendants.values())
        projections = sum(kind in {"projection", "lm_head"} for kind in descendants.values())
        heads = any(type(record["attributes"].get(key)) is int for key in
                    ("num_heads", "num_attention_heads", "n_heads", "head_dim"))
        if native and "attention" in class_name and projections >= 2 and heads:
            return result("attention", "Native attention class, head metadata and projection subtree", 0.98)
        if native and ("mlp" in class_name or "feedforward" in class_name or "ffn" in class_name) and projections >= 2:
            return result("mlp", "Native feed-forward class and multiple affine projections", 0.95)
        has_norm = bool(kinds & {"layer_norm", "rms_norm", "normalization"})
        has_stack = any(self.raw["modules"][p]["container"] == "module_list" for p in descendants)
        if native and "attention" in kinds and has_norm and projections >= 4 and not has_stack and "embedding" not in kinds:
            return result("transformer_block", "Attention, normalization and affine subtrees without a nested layer stack", 0.95)
        if path == self.raw["base_model_prefix"] and children:
            return result("backbone", "Native base_model_prefix identifies the backbone")
        return result("module", "Preserved actual module identity; semantic role not established")

    def is_leaf(self, path: str) -> bool:
        record = self.raw["modules"][path]
        return not record["children"] or self.classifications[path].kind in {
            "projection", "lm_head", "classifier", "embedding", "position_embedding", "layer_norm", "batch_norm",
            "rms_norm", "normalization", "dropout", "convolution", "activation", "pooling", "identity", "reshape",
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "classifications": {path: asdict(item) for path, item in self.classifications.items()},
            "unclassified_roles": [p for p, c in self.classifications.items() if c.kind == "module"],
            "scope": "Registered module roles; no expression interpretation or computation equivalence claim",
        }
