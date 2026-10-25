"""Family-specific validation for nested CLAP audio configuration schedules."""

import pytest

from model_anatomy.src import limits, registry
from model_anatomy.src.errors import ArtifactError, UnsupportedModelError

CLAP_CONFIG = {
    "model_type": "clap",
    "audio_config": {
        "model_type": "clap_audio_model",
        "depths": [2, 2, 6, 2],
        "num_hidden_layers": 4,
        "num_attention_heads": [4, 8, 16, 32],
    },
    "text_config": {
        "model_type": "clap_text_model",
        "num_attention_heads": 12,
    },
}


def test_real_clap_audio_schedule_is_accepted():
    config = registry.load_native_config(CLAP_CONFIG)

    assert list(config.audio_config.depths) == [2, 2, 6, 2]
    assert config.audio_config.num_hidden_layers == 4
    assert list(config.audio_config.num_attention_heads) == [4, 8, 16, 32]
    assert config.text_config.num_attention_heads == 12


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("depths", [2, 2, 6, 2]),
        ("num_hidden_layers", 4),
        ("num_attention_heads", [4, 8, 16, 32]),
    ],
)
def test_omitted_nested_clap_schedule_field_uses_installed_default(field, expected):
    source = {
        **CLAP_CONFIG,
        "audio_config": {key: value for key, value in CLAP_CONFIG["audio_config"].items() if key != field},
    }

    config = registry.load_native_config(source)

    value = getattr(config.audio_config, field)
    if isinstance(expected, list):
        assert list(value) == expected
    else:
        assert value == expected
    assert field not in source["audio_config"]


def test_all_omitted_nested_and_direct_clap_audio_schedules_use_installed_defaults():
    registry._validate_embedded_config({"model_type": "clap_audio_model"})
    source = {"model_type": "clap", "audio_config": {"model_type": "clap_audio_model"}}

    config = registry.load_native_config(source)

    assert list(config.audio_config.depths) == [2, 2, 6, 2]
    assert config.audio_config.num_hidden_layers == 4
    assert list(config.audio_config.num_attention_heads) == [4, 8, 16, 32]
    assert source == {"model_type": "clap", "audio_config": {"model_type": "clap_audio_model"}}


@pytest.mark.parametrize("field", ["depths", "num_hidden_layers", "num_attention_heads"])
def test_explicit_none_does_not_fall_back_to_clap_schedule_default(monkeypatch, field):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for an explicit null schedule field"),
    )
    config = {
        **CLAP_CONFIG,
        "audio_config": {**CLAP_CONFIG["audio_config"], field: None},
    }

    with pytest.raises(ArtifactError, match=f"CLAP audio stage schedule: {field}"):
        registry.load_native_config(config)


def test_installed_clap_schedule_defaults_obey_lowered_budget_before_autoconfig(monkeypatch):
    monkeypatch.setattr(limits, "MAX_MODEL_LAYERS", 10)
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run beyond the effective default schedule budget"),
    )

    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config({"model_type": "clap", "audio_config": {"model_type": "clap_audio_model"}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("depths", []),
        ("depths", [2, True, 6, 2]),
        ("depths", [2, 0, 6, 2]),
        ("depths", [2, -1, 6, 2]),
        ("depths", "2,2,6,2"),
        ("num_hidden_layers", 3),
        ("num_hidden_layers", True),
        ("num_attention_heads", [4, 8, 16]),
        ("num_attention_heads", [4, 8, 16, 0]),
        ("num_attention_heads", [4, 8, 16, True]),
    ],
)
def test_malformed_clap_audio_schedules_are_rejected_before_autoconfig(monkeypatch, field, value):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for a malformed CLAP schedule"),
    )
    config = {
        **CLAP_CONFIG,
        "audio_config": {**CLAP_CONFIG["audio_config"], field: value},
    }

    with pytest.raises(ArtifactError, match="CLAP audio stage schedule"):
        registry.load_native_config(config)


def test_clap_audio_stage_depth_and_head_budgets_are_enforced_before_autoconfig(monkeypatch):
    monkeypatch.setattr(limits, "MAX_MODEL_LAYERS", 10)
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run beyond CLAP schedule budgets"),
    )
    config = {
        **CLAP_CONFIG,
        "audio_config": {**CLAP_CONFIG["audio_config"], "depths": [2, 2, 6, 2]},
    }

    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config(config)

    config["audio_config"] = {
        **CLAP_CONFIG["audio_config"],
        "depths": [2, 2, 2, 2],
        "num_attention_heads": [4, 8, 16, 11],
    }
    with pytest.raises(ArtifactError, match="CLAP audio stage schedule: num_attention_heads"):
        registry.load_native_config(config)


def test_head_schedules_remain_rejected_outside_clap_audio():
    with pytest.raises(ArtifactError, match="num_attention_heads"):
        registry.load_native_config(
            {"model_type": "gpt2", "num_attention_heads": [4, 8], "n_embd": 16, "n_layer": 2, "n_head": 2}
        )
    config = {
        **CLAP_CONFIG,
        "text_config": {"model_type": "clap_text_model", "num_attention_heads": [4, 8]},
    }
    with pytest.raises(ArtifactError, match="num_attention_heads"):
        registry.load_native_config(config)


@pytest.mark.parametrize(
    "guard",
    [{"auto_map": {"AutoConfig": "publisher.Config"}}, {"use_pretrained_backbone": True}],
)
def test_nested_clap_guards_remain_rejected(monkeypatch, guard):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run when a nested guard fails"),
    )
    config = {**CLAP_CONFIG, "audio_config": {**CLAP_CONFIG["audio_config"], **guard}}

    with pytest.raises(UnsupportedModelError):
        registry.load_native_config(config)
