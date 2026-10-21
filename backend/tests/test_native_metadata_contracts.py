from __future__ import annotations

from types import SimpleNamespace

from torch import nn
from transformers import PretrainedConfig, T5Config

from model_anatomy.src import native_metadata_contracts
from model_anatomy.src.native_metadata_contracts import native_config_attribute
from model_anatomy.src.static_flow import StaticFlowAnalyzer


def test_attention_implementation_reads_only_reviewed_backing_field() -> None:
    config = T5Config()
    vars(config)["_attn_implementation_internal"] = "sdpa"

    assert native_config_attribute(config, "_attn_implementation") == (True, "sdpa")


def test_missing_or_unreviewed_config_metadata_stays_unknown() -> None:
    config = T5Config()
    assert native_config_attribute(config, "_attn_implementation") == (False, None)
    assert native_config_attribute(config, "model_type") == (False, None)


def test_source_fingerprint_mismatch_disables_contract(monkeypatch) -> None:
    config = T5Config()
    vars(config)["_attn_implementation_internal"] = "sdpa"
    monkeypatch.setitem(
        native_metadata_contracts.CONTRACT_SOURCE_SHA256,
        "transformers.configuration_utils",
        "0" * 64,
    )

    assert native_config_attribute(config, "_attn_implementation") == (False, None)


def test_overridden_property_is_not_invoked() -> None:
    class UnsafeConfig(PretrainedConfig):
        @property
        def _attn_implementation(self):
            raise AssertionError("descriptor must not run")

    config = object.__new__(UnsafeConfig)
    assert native_config_attribute(config, "_attn_implementation") == (False, None)


class ConfigBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = T5Config()
        vars(self.config)["_attn_implementation_internal"] = "sdpa"

    def forward(self, x):
        if self.config._attn_implementation == "sdpa":
            return x
        return x


def test_static_flow_resolves_reviewed_config_property_without_warning() -> None:
    module = ConfigBranch()
    classifications = {name: SimpleNamespace(kind="backbone") for name, _ in module.named_modules()}
    flow = StaticFlowAnalyzer(module.eval(), "", classifications).analyze()

    assert not any("Conditional source path is unresolved" in warning for warning in flow.warnings)
    assert any(item["code"] == "native_config_attribute" for item in flow.diagnostics)
