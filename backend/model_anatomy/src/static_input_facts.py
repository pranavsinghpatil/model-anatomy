"""Internal static facts used to analyze model source without executing it."""

from __future__ import annotations

import torch

from .ir import TensorSpec


class InputResolver:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.native_vit = (
            getattr(model.config, "model_type", None) == "vit"
            and type(model).__module__ == "transformers.models.vit.modeling_vit"
        )
        native_module = type(model).__module__
        model_type = getattr(model.config, "model_type", None)
        self.native_image_text = (
            model_type == "clip" and native_module == "transformers.models.clip.modeling_clip"
        ) or (model_type == "siglip" and native_module == "transformers.models.siglip.modeling_siglip")
        parameter = next(model.parameters(), None)
        self.activation_dtype = {
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.float64: "F64",
        }.get(parameter.dtype if parameter is not None else None)

    def spec(self, name: str, path: str = "") -> TensorSpec:
        config = self.model.config
        sequence = (
            "T" if name == "decoder_input_ids" or "decoder" in path and config.is_encoder_decoder else "S"
        )
        if name in {"encoder_hidden_states", "key_value_states"}:
            sequence = "S"
        hidden = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if self.native_vit:
            size, patch = config.image_size, config.patch_size
            size = size if isinstance(size, (list, tuple)) else (size, size)
            patch = patch if isinstance(patch, (list, tuple)) else (patch, patch)
            if all(type(v) is int and v > 0 for v in (*size, *patch)):
                sequence = (size[0] // patch[0]) * (size[1] // patch[1]) + 1
            module = self.model.get_submodule(path) if path else self.model
            if name == "hidden_states" and type(module).__name__ == "ViTOutput":
                hidden = module.dense.in_features
        heads = getattr(config, "num_attention_heads", None) or getattr(config, "num_heads", None)
        if name == "position_bias":
            module = self.model.get_submodule(path) if path else self.model
            native_t5_cross = (
                getattr(config, "model_type", None) == "t5"
                and type(self.model).__module__ == "transformers.models.t5.modeling_t5"
                and "EncDecAttention" in path.split(".")
            )
            cross = (
                "cross" in type(module).__name__.lower()
                or "encdec" in type(module).__name__.lower()
                or native_t5_cross
            )
            return TensorSpec(
                [1, heads, sequence, "S" if cross else sequence] if heads else None,
                self.activation_dtype,
                name,
            )
        if name == "position_embeddings":
            dimension = getattr(config, "head_dim", None) or (hidden // heads if hidden and heads else None)
            return TensorSpec(
                ["B", sequence, dimension] if dimension else None, self.activation_dtype, "rotary_cos_sin"
            )
        if name in {"input_ids", "decoder_input_ids", "token_type_ids"}:
            return TensorSpec(["B", sequence], "I64", name)
        if name == "position_ids":
            return TensorSpec([1, sequence], "I64", name)
        if name == "pixel_values":
            from transformers.models.auto.modeling_auto import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES

            model_type = getattr(config, "model_type", None)
            native_module = type(self.model).__module__
            native_videomae = (
                model_type == "videomae" and native_module == "transformers.models.videomae.modeling_videomae"
            )
            if model_type in MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES and not native_videomae:
                return TensorSpec(None, self.activation_dtype, name)
            image_config = config
            if getattr(config, "vision_config", None) is not None:
                if self.native_image_text:
                    image_config = config.vision_config
                else:
                    # Composite models can pack patches rather than use image axes.
                    return TensorSpec(None, self.activation_dtype, name)
            size = getattr(image_config, "image_size", None)
            shape = (
                list(size)
                if isinstance(size, (list, tuple))
                else [size, size]
                if type(size) is int
                else ["H", "W"]
            )
            axes = ["B", getattr(image_config, "num_channels", 3), *shape]
            if native_videomae:
                # Native VideoMAEPatchEmbeddings.forward unpacks B,T,C,H,W.
                frames = getattr(config, "num_frames", None)
                axes.insert(1, frames if type(frames) is int and frames > 0 else "Frames")
            return TensorSpec(axes, self.activation_dtype, name)
        if name in {"pixel_values_videos", "video_pixel_values"}:
            return TensorSpec(None, self.activation_dtype, name)
        if name == "input_features":
            return TensorSpec(
                ["B", getattr(config, "num_mel_bins", "F"), "Frames"], self.activation_dtype, name
            )
        if name == "input_values":
            return TensorSpec(["B", "Samples"], self.activation_dtype, name)
        return (
            TensorSpec(["B", sequence, hidden], self.activation_dtype, name)
            if type(hidden) is int
            else TensorSpec(None, None, name)
        )
