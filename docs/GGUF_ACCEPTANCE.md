# GGUF Compatibility Acceptance

## User-Reported Failure

Analysis `8112b8e4cda147f9b3dd23560b04bd62` inspected only the file inventory of
`prism-ml/Ternary-Bonsai-2-27B-gguf` using engine `2026.10.12.5`. It had no tensor
records and reported missing architecture configuration. That was a real format
gap, not evidence that Bonsai itself was invalid.

Pinned revision: `b072e1d3b35a0a630cece372c2127528e0994386`.
Its files include F16, PQ2_0 and PTQ1_0 language-model variants, plus BF16 and Q8_0
vision projectors. These are alternatives/components, not interchangeable shards.

## Manual Check

1. Open `http://127.0.0.1:5173/` and submit the repository ID again. Old reports
   remain unchanged; the new engine has a different cache identity.
2. Confirm the result asks for checkpoint selection rather than choosing a
   variant or combining all five GGUF files.
3. Select the inspect icon beside `Ternary-Bonsai-2-27B-PTQ1_0.gguf`.
4. Check the pinned revision, tensor names, GGML-order dimensions, numeric
   quantization IDs and evidence. Inspect subsequent pages and export JSON/audit.
5. Confirm there is no invented computation graph, module ownership or unique
   model parameter claim. Tensor element totals describe the selected artifact.

The exact pinned file URL can also be submitted directly:

```text
https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

## Boundaries

This increment inspects GGUF headers, not Bonsai's execution graph. GGUF metadata
is self-described; headers do not verify tensor payload integrity, dequantization
correctness or inference behavior. Unknown quantization identifiers remain
numeric facts rather than guessed encodings. Oversized metadata is bounded;
malformed headers fail validation. Split files must not be presented as a fully
inspected model. Publisher code and model weights are never loaded.

Architecture support across encoder-decoder, multimodal, audio and video families
still requires the separate acceptance gates in `SYSTEM_GAPS.md`.

The zero-payload audit is established only after a successful header parse and
range-boundary validation. A corrupt byte stream cannot prove where its real
tensor data begins; an interrupted/budget-limited GGUF parse does not publish a
zero-payload claim. Retained scalar metadata and tensor descriptors are checked;
skipped tokenizer strings are hashed but not decoded or semantically validated.

## Recorded Evidence (October 12)

Engine `2026.10.12.6`, local SQLite/storage and the real inspection worker:

- Bonsai PTQ1_0: **851 tensors**, **26,895,998,464 selected-file elements**,
  49 metadata entries, file size **5,946,648,928 bytes**. The elements are not
  labelled unique model parameters.
- **43 range requests**, **11,121,006 bytes transferred**, including a repeated
  24-byte prefix; parsed header length **11,120,982**, aligned data start
  **11,120,992**. All reads ended before that data section; zero payload bytes.
- Numeric tensor type IDs are `0`, `30`, and publisher-specific `143`. Upstream
  names are only interpretations; no ternary decoding or payload validation runs.
- Real desktop/mobile repository submission, explicit file selection, persisted
  tensor result and exports passed. The cold browser workflow took about one
  minute; the second visitor reused the revision/file/version-specific cache.
- A separate pinned `bartowski/SmolLM2-135M-Instruct-GGUF` Q4_K_M check at
  `09816acd5d99df7be770d85ea30822623dab342c` returned **272 tensors**. The final
  direct metadata profile used **30 ranges / 1,788,958 bytes / 26.17 seconds**.
  These timings are local observations, not hosted latency guarantees.

Earlier reader candidates were too close to or exceeded the 120-second worker
deadline. The final reader uses bounded structural-minimum buffering across
future metadata/tensor records; regressions include minimal descriptors and
large tokenizer arrays, complete parsed-header hashing and split-file scope.
The independent early review identified malformed-boundary and split reporting
risks, now covered by explicit guards/tests. Its final follow-up was unavailable
because the review agent reached its usage limit; parent diff review and
deterministic verification were completed, but an independent final approval is
still a PR review gate.

Final verification: **276 backend/infrastructure tests passed** (11 opt-in or
platform skips), **70 deterministic browser tests passed** (12 live checks
disabled in that run), and **all 12 isolated live desktop/mobile checks passed**.
The live suite also rechecked Kolibri, Qwen Image LoRA, SmolLM2 and ViT. Build,
TypeScript, actual generated OpenAPI types, formatting, Ruff, compileall and
dependency-lock consistency passed. An earlier overlapping browser run failed
while saving traces because two suites shared an output directory; the isolated
rerun above passed and is the recorded final evidence.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_gguf_inspection.py backend/tests/test_gguf_references.py -q
$env:E2E_BASE_URL = 'http://127.0.0.1:5173'
$env:E2E_LIVE_API = '1'
Set-Location frontend
npm run test:e2e -- e2e/live-gguf.spec.ts --workers=2
```
