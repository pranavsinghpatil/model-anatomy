# Native ViT Static Architecture Acceptance

## Scope

Engine `2026.10.09.1` closes the canonical native ViT classification path without
calling forward or reading checkpoint tensor payloads. It preserves the report
schema and CLI. Earlier reports remain readable; the cache version changes so
new analyses cannot reuse earlier extraction results.

Pinned checkpoint: `google/vit-base-patch16-224` at
`10064cc17b69d623a025c9e90e11230cc8ba6967`.

| Step | Shape |
| --- | --- |
| Pixel input | `[B,3,224,224]` |
| Patch convolution | `[B,768,14,14]` |
| Flatten and transpose | `[B,196,768]` |
| Expand CLS, concatenate patches, add positions | `[B,197,768]` |
| Encoder and output projections/residuals | `[B,197,768]` |
| Select CLS token | `[B,768]` |
| Classification logits | `[B,1000]` |

Source transformations have native-source fingerprints and line evidence. Output
modules are classified from native implementation plus affine/dropout topology;
residual edges come from their actual source, not the class label. Parameter
nodes expose their own checkpoint tensors and counts, not the whole embedding
module. Invalid or unsupported shape operations retain unknown dimensions.

## Manual Check

1. Open the running local app at `http://127.0.0.1:5173/`.
2. Submit the pinned URL below with **Analyze Model**, not a prepared example.
3. At the root, inspect **Select tokens** and **Classifier**: the former changes
   `[B,197,768]` to `[B,768]`; the latter produces `[B,1000]`.
4. Open **Backbone**, **Embeddings**, then **Patch Embedding** to inspect the
   convolution, flatten and transpose. In Embeddings, inspect `cls_token` and
   `position_embeddings` under Tensors and Evidence.
5. Download Architecture JSON or Complete ZIP and compare the shapes above.

```text
https://huggingface.co/google/vit-base-patch16-224/blob/10064cc17b69d623a025c9e90e11230cc8ba6967/model.safetensors
```

Static inference is not execution proof, even when the canonical report completes.
This application inspects model structure; it does not classify an uploaded image.
Image preprocessing, masking, interpolated resolutions, other vision families,
video and multimodal bridges are not established by this acceptance case.

## Verification

Committed checks cover tiny square and nonsquare ViT graphs, bounded shape algebra
including unknown/invalid transformations, the pinned real checkpoint, and a
fresh real HTTP-worker cold/cache analysis with ownership, exports and isolation.
`frontend/e2e/live-vit.spec.ts` uses the real service without request interception
on desktop and mobile, submits the pinned model, verifies the engine version and
CLS/classifier shapes, opens nested patch graphs, checks classifier tensors and
downloads ZIP. It needs `E2E_LIVE_API=1` and the API/worker on port 8001.

Final local run: 128 backend tests passed with real HF and HTTP integration
enabled, one Linux-only memory test skipped and one existing Starlette TestClient
deprecation warning. All 48 browser checks passed (46 existing checks plus two
real ViT desktop/mobile submissions). Frontend build/type check and formatting,
backend Ruff, and scoped diff checks passed. Independent review requested explicit
residual connectivity assertions; those now verify both operands, skip ownership,
output connectivity and the absence of a residual inside `ViTSelfOutput`.

Cloud deployment, real Supabase operation and live Nebius/NVIDIA explanations
remain independent gates. Local verification does not establish those systems.

## QKV Export Correction

Engine `2026.10.09.2` corrects three Q/K/V projection-output edges which lost
already-established shapes when assigned named graph ports. This was a graph
export defect, not an inherent unknown in ViT. The pinned canonical ViT now
requires full shape coverage and `completed` status in real HTTP acceptance.
General static-analysis limitations remain visible; completed does not mean all
runtime variants or arbitrary models are verified. Previously saved reports are
immutable: submit a new analysis to obtain the corrected version.

Correction verification: full backend suite 128 passed, one Linux-only skip and
the existing deprecation warning; then four focused offline ViT tests passed,
including the added QKV export regression. Real desktop/mobile checks both pass
with `completed` required. Initial browser rerun detected the still-running old
API serving its previous cache version; API and worker were restarted before
the successful rerun. Ruff, formatting and scoped diff checks pass.
