"""Retired activation contracts must fail closed under the pinned native source.

The current report is a registered-module architecture. These older helpers are
kept as explicit unknowns until their source rules receive a separate review.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import transformers
from transformers.activations import GELUActivation

from model_anatomy.src.ir import TensorSpec
from model_anatomy.src.native_output_contracts import (
    CONTRACT_SOURCE_SHA256,
    native_module_outputs,
)


def test_pinned_transformers_source_has_not_been_silently_reblessed():
    assert transformers.__version__ == "5.20.2"
    drifted = []
    for name, reviewed_hash in CONTRACT_SOURCE_SHA256.items():
        module = importlib.import_module(name)
        actual_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual_hash != reviewed_hash:
            drifted.append(name)
    assert set(drifted) == set(CONTRACT_SOURCE_SHA256)


def test_stale_native_activation_contract_does_not_claim_an_output():
    module = GELUActivation()
    input_spec = TensorSpec(["B", "S", 16], "F32", "hidden_states")

    assert native_module_outputs(module, input_spec) is None


def test_replaced_or_unreviewed_native_implementation_stays_unknown(monkeypatch):
    module = GELUActivation()
    input_spec = TensorSpec(["B", "S", 16], "F32", "hidden_states")
    monkeypatch.setitem(CONTRACT_SOURCE_SHA256, type(module).__module__, "0" * 64)

    assert native_module_outputs(module, input_spec) is None
    assert native_module_outputs(GELUActivation(), input_spec) is None
