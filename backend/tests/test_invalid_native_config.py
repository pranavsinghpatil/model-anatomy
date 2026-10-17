import pytest

from model_anatomy.src.errors import ArtifactError
from model_anatomy.src.registry import instantiate_native, load_native_config


def test_incompatible_positive_native_dimensions_are_invalid_not_unsupported():
    config = load_native_config(
        {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 32,
            "n_positions": 16,
            "n_embd": 9,
            "n_head": 2,
            "n_layer": 1,
        }
    )
    with pytest.raises(ArtifactError, match="Invalid native model configuration"):
        instantiate_native(config)


def test_known_native_config_constructor_failure_is_not_unknown_architecture(monkeypatch):
    from model_anatomy.src import registry

    def malformed(*args, **kwargs):
        raise ValueError("malformed constructor fields")

    monkeypatch.setattr(registry.AutoConfig, "for_model", malformed)
    with pytest.raises(ArtifactError, match="Invalid native configuration"):
        load_native_config({"model_type": "gpt2"})


def test_web_fallback_cannot_hide_invalid_native_dimensions():
    from model_anatomy.src.artifacts import CheckpointMetadata, MetadataClient
    from model_anatomy.src.pipeline import reconstruct

    class Client(MetadataClient):
        def resolve(self, reference):
            revision = "b" * 40
            self.repository_files[(reference.model_id, revision)] = {"config.json", "model.safetensors"}
            return revision

        def json(self, url, max_bytes=1024 * 1024):
            return {
                "model_type": "gpt2",
                "architectures": ["GPT2LMHeadModel"],
                "vocab_size": 32,
                "n_positions": 16,
                "n_embd": 9,
                "n_head": 2,
                "n_layer": 1,
            }

        def checkpoint(self, reference, revision):
            return CheckpointMetadata({}, {}, {})

    with pytest.raises(ArtifactError, match="Invalid native model configuration"):
        reconstruct("publisher/model", client=Client(), allow_artifact_inspection=True)
