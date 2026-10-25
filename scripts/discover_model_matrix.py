"""Create the pinned, metadata-only Hugging Face model benchmark matrix."""

from __future__ import annotations

import concurrent.futures
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# These are repository candidates, not claims about successful reconstruction.
# Every accepted ID, task, revision, and optional component path is checked via
# the public Hugging Face model API before it enters the manifest.
CANDIDATES: list[tuple[str, str, str | None, str | None]] = [
    # Causal language models, code, reasoning, and preference/reward artifacts
    ("HuggingFaceTB/SmolLM2-135M", "causal-lm", None, None),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "causal-lm", None, None),
    ("Qwen/Qwen3-0.6B", "causal-lm", None, None),
    ("Qwen/Qwen2.5-0.5B-Instruct", "causal-lm", None, None),
    ("openai-community/gpt2", "causal-lm", None, None),
    ("EleutherAI/pythia-70m", "causal-lm", None, None),
    ("EleutherAI/pythia-160m", "causal-lm", None, None),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "causal-lm", None, None),
    ("microsoft/phi-2", "causal-lm", None, None),
    ("microsoft/Phi-3-mini-4k-instruct", "causal-lm", None, None),
    ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "code-lm", None, None),
    ("bigcode/starcoder2-3b", "code-lm", None, None),
    ("deepseek-ai/deepseek-coder-1.3b-instruct", "code-lm", None, None),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "causal-lm", None, None),
    ("mistralai/Mistral-7B-Instruct-v0.3", "causal-lm", None, "gated; API metadata is public but artifacts may require access"),
    ("meta-llama/Llama-3.2-1B-Instruct", "causal-lm", None, "gated; API metadata is public but artifacts may require access"),
    ("HuggingFaceH4/zephyr-7b-beta", "rlhf-sft-lm", None, "SFT/chat checkpoint; RLHF training provenance is a repository declaration"),
    ("OpenAssistant/reward-model-deberta-v3-large", "rlhf-reward-model", None, "Reward model, not a serialized RL policy"),
    ("RLHFlow/ArmoRM-Llama3-8B-v0.1", "rlhf-reward-model", None, "Reward model, not a serialized RL policy"),
    ("OpenAssistant/oasst-sft-1-pythia-12b", "rlhf-sft-lm", None, "SFT/chat checkpoint; large metadata-only case"),
    # Text encoders, embeddings, reranking, question answering, classification
    ("sentence-transformers/all-MiniLM-L6-v2", "embedding", None, None),
    ("BAAI/bge-small-en-v1.5", "embedding", None, None),
    ("intfloat/e5-small-v2", "embedding", None, None),
    ("thenlper/gte-small", "embedding", None, None),
    ("mixedbread-ai/mxbai-embed-xsmall-v1", "embedding", None, None),
    ("BAAI/bge-base-en-v1.5", "embedding", None, None),
    ("intfloat/e5-base-v2", "embedding", None, None),
    ("sentence-transformers/paraphrase-MiniLM-L3-v2", "embedding", None, None),
    ("jinaai/jina-embeddings-v2-small-en", "embedding", None, None),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker", None, None),
    ("BAAI/bge-reranker-base", "reranker", None, None),
    ("deepset/roberta-base-squad2", "extractive-qa", None, None),
    ("distilbert/distilbert-base-uncased-distilled-squad", "extractive-qa", None, None),
    ("google-bert/bert-base-uncased", "text-classification-encoder", None, None),
    ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "sentiment-classification", None, None),
    ("SamLowe/roberta-base-go_emotions", "text-classification", None, None),
    ("unitary/toxic-bert", "text-classification", None, None),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "sentiment-classification", None, None),
    ("FacebookAI/roberta-base", "text-classification-encoder", None, None),
    # Encoder-decoder and translation/summarization
    ("google-t5/t5-small", "seq2seq", None, None),
    ("google/flan-t5-small", "seq2seq-instruct", None, None),
    ("google/flan-t5-base", "seq2seq-instruct", None, None),
    ("sshleifer/distilbart-cnn-12-6", "summarization", None, None),
    ("facebook/bart-base", "seq2seq", None, None),
    ("Helsinki-NLP/opus-mt-en-fr", "translation", None, None),
    ("Helsinki-NLP/opus-mt-fr-en", "translation", None, None),
    ("Helsinki-NLP/opus-mt-en-de", "translation", None, None),
    ("facebook/nllb-200-distilled-600M", "translation", None, None),
    ("google/pegasus-xsum", "summarization", None, None),
    # Vision classification, detection, segmentation, and depth
    ("google/vit-base-patch16-224", "image-classification", None, None),
    ("timm/mobilenetv3_small_100.lamb_in1k", "image-classification", None, None),
    ("microsoft/resnet-50", "image-classification", None, None),
    ("facebook/convnext-tiny-224", "image-classification", None, None),
    ("google/vit-base-patch16-224-in21k", "image-classification", None, None),
    ("microsoft/swin-tiny-patch4-window7-224", "image-classification", None, None),
    ("nvidia/mit-b0", "image-classification", None, None),
    ("facebook/detr-resnet-50", "object-detection", None, None),
    ("hustvl/yolos-tiny", "object-detection", None, None),
    ("microsoft/table-transformer-detection", "object-detection", None, None),
    ("microsoft/table-transformer-structure-recognition", "object-detection", None, None),
    ("nvidia/segformer-b0-finetuned-ade-512-512", "image-segmentation", None, None),
    ("nvidia/segformer-b1-finetuned-ade-512-512", "image-segmentation", None, None),
    ("facebook/mask2former-swin-tiny-coco-instance", "image-segmentation", None, None),
    ("CIDAS/clipseg-rd64-refined", "image-segmentation", None, None),
    ("depth-anything/Depth-Anything-V2-Small-hf", "depth-estimation", None, None),
    ("Intel/dpt-hybrid-midas", "depth-estimation", None, None),
    ("facebook/sam-vit-base", "image-segmentation", None, None),
    # Image-text, multimodal, and video understanding
    ("openai/clip-vit-base-patch32", "image-text-embedding", None, None),
    ("google/siglip-base-patch16-224", "image-text-embedding", None, None),
    ("Salesforce/blip-image-captioning-base", "image-captioning", None, None),
    ("Salesforce/blip-vqa-base", "visual-question-answering", None, None),
    ("dandelin/vilt-b32-finetuned-vqa", "visual-question-answering", None, None),
    ("HuggingFaceTB/SmolVLM-256M-Instruct", "image-text-to-text", None, None),
    ("Qwen/Qwen2-VL-2B-Instruct", "image-text-to-text", None, None),
    ("llava-hf/llava-1.5-7b-hf", "image-text-to-text", None, "large metadata-only case"),
    ("naver-clova-ix/donut-base-finetuned-cord-v2", "document-understanding", None, None),
    ("google/pix2struct-base", "image-text-to-text", None, None),
    ("MCG-NJU/videomae-base", "video-classification", None, None),
    ("facebook/timesformer-base-finetuned-k400", "video-classification", None, None),
    ("microsoft/xclip-base-patch32", "video-text-retrieval", None, None),
    # Speech recognition, TTS, conversion, and speech-to-speech
    ("openai/whisper-tiny", "speech-recognition", None, None),
    ("openai/whisper-base", "speech-recognition", None, None),
    ("facebook/wav2vec2-base-960h", "speech-recognition", None, None),
    ("facebook/hubert-base-ls960", "speech-representation", None, None),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-english", "speech-recognition", None, None),
    ("facebook/mms-1b-all", "speech-recognition", None, None),
    ("facebook/seamless-m4t-v2-small", "speech-translation", None, "multitask speech translation; inspect task modes before treating as speech-to-speech"),
    ("microsoft/speecht5_tts", "text-to-speech", None, None),
    ("microsoft/speecht5_vc", "voice-conversion", None, "voice conversion component; not an end-to-end duplex assistant"),
    ("myshell-ai/OpenVoice", "voice-conversion", None, "voice conversion; distinguish from speech-to-speech translation"),
    ("coqui/XTTS-v2", "voice-cloning-tts", None, "voice-cloning text-to-speech; not speech-to-speech conversion"),
    ("facebook/encodec_24khz", "neural-audio-codec", None, None),
    ("facebook/mms-tts-eng", "text-to-speech", None, None),
    ("suno/bark-small", "text-to-speech", None, None),
    ("parler-tts/parler-tts-mini-v1", "text-to-speech", None, None),
    ("hexgrad/Kokoro-82M", "text-to-speech", None, None),
    ("kyutai/hibiki-2b-pytorch-bf16", "speech-to-speech-translation", None, "native speech-to-speech translation checkpoint"),
    ("kyutai/hibiki-2b-pytorch-bf16", "speech-to-speech-translation", None, "native speech-to-speech translation checkpoint"),
    ("kyutai/tts-1.6b-en_fr", "text-to-speech", None, None),
    # Text/image generation and video generation; component references are
    # selected only after checking the repository's actual sibling filenames.
    ("hf-internal-testing/tiny-stable-diffusion-pipe", "text-to-image", "unet/diffusion_pytorch_model.safetensors", None),
    ("segmind/tiny-sd", "text-to-image", "unet/diffusion_pytorch_model.safetensors", None),
    ("stabilityai/sd-turbo", "text-to-image", "unet/diffusion_pytorch_model.safetensors", "gated/access restrictions may apply"),
    ("stabilityai/sdxl-turbo", "text-to-image", "unet/diffusion_pytorch_model.safetensors", "large metadata-only case"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "text-to-image", "unet/diffusion_pytorch_model.safetensors", "large metadata-only case; gated/access restrictions may apply"),
    ("stabilityai/stable-diffusion-2-1-base", "text-to-image", "unet/diffusion_pytorch_model.safetensors", None),
    ("black-forest-labs/FLUX.1-schnell", "text-to-image", "transformer/diffusion_pytorch_model.safetensors", "large metadata-only case; gated/access restrictions may apply"),
    ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "text-to-image", "transformer/diffusion_pytorch_model.safetensors", "large metadata-only case"),
    ("DeepFloyd/IF-I-XL-v1.0", "text-to-image", "unet/diffusion_pytorch_model.safetensors", "gated/access restrictions may apply"),
    ("Tencent-Hunyuan/HunyuanVideo", "text-to-video", "transformer/diffusion_pytorch_model.safetensors", "very large metadata-only case"),
    ("THUDM/CogVideoX-2b", "text-to-video", "transformer/diffusion_pytorch_model.safetensors", "large metadata-only case"),
    ("Lightricks/LTX-Video", "text-to-video", "transformer/diffusion_pytorch_model.safetensors", "large metadata-only case"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "text-to-video", "transformer/diffusion_pytorch_model.safetensors", "large metadata-only case"),
    ("cerspense/zeroscope_v2_576w", "text-to-video", "unet/diffusion_pytorch_model.safetensors", None),
    ("ali-vilab/i2vgen-xl", "image-to-video", "unet/diffusion_pytorch_model.safetensors", "large metadata-only case"),
    # Genuine RL policy serialization and adapter/quantized repositories.
    ("araffin/ppo-CartPole-v1", "rl-policy", None, "serialized Stable-Baselines3 policy; distinct from RLHF/reward-model artifacts"),
    ("sb3/ddpg-LunarLanderContinuous-v2", "rl-policy", None, "serialized Stable-Baselines3 policy; distinct from RLHF/reward-model artifacts"),
    ("edbeeching/decision-transformer-gym-hopper-medium", "rl-policy-decision-transformer", None, "offline RL sequence policy"),
    ("edbeeching/decision-transformer-gym-walker2d-medium", "rl-policy-decision-transformer", None, "offline RL sequence policy"),
    ("edbeeching/decision-transformer-gym-halfcheetah-medium", "rl-policy-decision-transformer", None, "offline RL sequence policy"),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "gguf-quantized", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", None),
    ("TheBloke/phi-2-GGUF", "gguf-quantized", "phi-2.Q4_K_M.gguf", None),
    ("bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF", "gguf-quantized", "Q4_K_M.gguf", None),
    ("HuggingFaceTB/SmolLM2-135M-Instruct-GGUF", "gguf-quantized", None, None),
    ("microsoft/Phi-3-mini-4k-instruct-gguf", "gguf-quantized", "Q4_K_M.gguf", None),
    ("tloen/alpaca-lora-7b", "peft-adapter", None, "adapter repository; classification is based on adapter files/config"),
]


def get_model(model_id: str) -> dict[str, Any]:
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(model_id, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "model-anatomy-benchmark-discovery/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def inspect(candidate: tuple[str, str, str | None, str | None]) -> dict[str, Any] | None:
    model_id, category, requested_file, notes = candidate
    try:
        model = get_model(model_id)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"SKIP {model_id}: {exc}")
        return None
    revision = model.get("sha")
    if not isinstance(revision, str) or len(revision) != 40:
        print(f"SKIP {model_id}: no 40-character API revision")
        return None
    siblings = [entry.get("rfilename", "") for entry in model.get("siblings", [])]
    reference = None
    if requested_file:
        selected = requested_file if requested_file in siblings else None
        if selected is None:
            # The API file list is authoritative. Pin one specific diffusion
            # component or one GGUF quantization variant, never an arbitrary
            # collection of shards/variants.
            if requested_file.lower().endswith(".gguf"):
                gguf_files = sorted(filename for filename in siblings if filename.lower().endswith(".gguf"))
                for quant in ("q4_k_m", "q4_0", "q4", "q5_k_m", "q5", "q3_k_m", "q8_0"):
                    selected = next((filename for filename in gguf_files if quant in filename.lower()), None)
                    if selected:
                        break
                if selected is None and gguf_files:
                    selected = gguf_files[0]
            else:
                for filename in siblings:
                    lower = filename.lower()
                    if lower.endswith(".safetensors") and ("unet/" in lower or "transformer/" in lower):
                        selected = filename
                        break
        if selected:
            reference = f"https://huggingface.co/{model_id}/blob/{revision}/{selected}"
        else:
            print(f"NOTE {model_id}: requested component missing; keeping repository-level inventory case")
    result: dict[str, Any] = {
        "model_id": model_id,
        "category": category,
        "revision": revision,
    }
    if reference:
        result["reference"] = reference
    if model.get("pipeline_tag"):
        result["declared_task"] = model["pipeline_tag"]
    note_parts = [part for part in (notes, f"API library_name={model.get('library_name') or 'unspecified'}") if part]
    if note_parts:
        result["notes"] = "; ".join(note_parts)
    print(f"OK {model_id} [{category}] {model.get('pipeline_tag')} files={len(siblings)}")
    return result


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        inspected = list(pool.map(inspect, CANDIDATES))
    cases = [case for case in inspected if case]
    # Never count repeated repository IDs as independent model coverage.
    unique: dict[str, dict[str, Any]] = {}
    for case in cases:
        unique.setdefault(case["model_id"], case)
    manifest = {
        "schema_version": 1,
        "discovered_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "cases": list(unique.values()),
    }
    destination = Path(__file__).resolve().parents[1] / "benchmarks" / "model-matrix-100.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {destination} ({len(unique)} distinct repositories)")


if __name__ == "__main__":
    main()
