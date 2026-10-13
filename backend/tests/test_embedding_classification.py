"""An embedding task/model name must not collapse its Transformer backbone."""

from types import SimpleNamespace

import pytest
import torch
from test_engine import checkpoint_for

from model_anatomy.src.facts import inspect_native_model
from model_anatomy.src.patterns import SemanticAnalyzer


class EmbeddingBackbone(torch.nn.Module):
    def __init__(self, extra):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(16, 4, device="meta")
        if extra == "attention":
            self.attention = torch.nn.Module()
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(self.attention, name, torch.nn.Linear(4, 4, device="meta"))
        elif extra == "mlp":
            self.mlp = torch.nn.Module()
            self.mlp.fc1 = torch.nn.Linear(4, 8, device="meta")
            self.mlp.fc2 = torch.nn.Linear(8, 4, device="meta")

    def forward(self, input_ids):
        raise AssertionError("Classification must never execute forward")


class Root(torch.nn.Module):
    base_model_prefix = "core"

    def __init__(self, extra):
        super().__init__()
        self.config = SimpleNamespace(architectures=["EmbeddingBackbone"], is_encoder_decoder=False)
        self.core = EmbeddingBackbone(extra)


@pytest.mark.parametrize("extra", ["attention", "mlp"])
def test_embedding_named_transformer_uses_backbone_ownership_not_embedding_label(extra):
    model = Root(extra)
    raw = inspect_native_model(model, checkpoint_for(model))
    classification = SemanticAnalyzer(model, raw).classifications["core"]
    assert classification.kind == "backbone"
    assert any("base_model_prefix" in reason for reason in classification.evidence)


def test_actual_embedding_only_subtree_retains_its_embedding_label():
    model = Root(None)
    raw = inspect_native_model(model, checkpoint_for(model))
    assert SemanticAnalyzer(model, raw).classifications["core"].kind == "embeddings"
