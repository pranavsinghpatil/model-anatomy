"""Native metadata contracts stay unknown after unreviewed source drift."""

from __future__ import annotations

import torch
from transformers import PretrainedConfig, T5Config
from transformers.models.t5.modeling_t5 import T5Stack

from model_anatomy.src.native_metadata_contracts import (
    native_config_attribute,
    native_module_attribute,
    trusted_t5_stack_method,
    trusted_torch_tensor_method,
)


def _stack() -> T5Stack:
    config = T5Config(d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2)
    with torch.device("meta"):
        return T5Stack(config).eval()


def test_unreviewed_config_source_does_not_certify_attention_property():
    config = T5Config()
    vars(config)["_attn_implementation_internal"] = "sdpa"
    assert native_config_attribute(config, "_attn_implementation") == (False, None)


def test_untrusted_descriptor_is_not_invoked():
    class UnsafeConfig(PretrainedConfig):
        @property
        def _attn_implementation(self):
            raise AssertionError("descriptor must not run")

    config = object.__new__(UnsafeConfig)
    assert native_config_attribute(config, "_attn_implementation") == (False, None)


def test_unreviewed_t5_source_does_not_certify_dtype_or_mask_method():
    module = _stack()
    assert native_module_attribute(module, "dtype") == (False, None)
    assert not trusted_t5_stack_method(module, "forward", T5Stack.forward)


def test_tensor_method_identity_guard_detects_replacement(monkeypatch):
    assert trusted_torch_tensor_method("clone")
    monkeypatch.setattr(torch.Tensor, "clone", lambda self: self)
    assert not trusted_torch_tensor_method("clone")
