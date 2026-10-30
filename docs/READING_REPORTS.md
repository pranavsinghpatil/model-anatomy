# Reading and Testing a Model Anatomy Report

## Three Different Things

| Field | Plain meaning |
| --- | --- |
| Analysis status | Whether this inspection finished, found gaps, or failed. |
| Model revision | The exact Hugging Face artifact version inspected. It is not a status. |
| Engine version | The version of Model Anatomy's extraction rules used for this report. |

The same model ID can resolve to a newer revision later. To reproduce a report,
use its recorded revision rather than assuming the repository's latest files
are unchanged. A cached report reuses a real inspection with the same immutable
revision and extraction identity. Prepared examples and their API have been
removed; only submitted, visitor-owned analyses are available in the application.

## What Completed Means

The recorded canonical static analysis has complete component/module
classification, checkpoint parameter mapping and graph connection shapes, with
no extraction warnings or reported unknown components/unmapped tensors.

**100% coverage is not a probability of correctness or proof of every runtime
behavior.** It measures how much of the inspected representation has established
facts. Model Anatomy does not run model inference or validate numerical weight
values. Direct tensor metadata, reconstructed connections and their evidence
remain distinguishable even in a completed report.

Partial means an inspection produced a usable report but still has an actual
extraction warning or incomplete/unrecorded coverage. Check **Incomplete checks**
and **Warnings** for its specific reasons. Unsupported means the artifact or
architecture cannot be inspected safely under current policy. Failed, timed out
and cancelled are unsuccessful jobs, not substitute model reports.

## What Artifacts Inspected Means

**Artifacts inspected** is not a completed architecture and is not a renamed
partial report. The repository was pinned and its actual files were inspected,
but the application cannot safely construct its architecture. The page states
why, and shows verified tensor names, shapes, dtypes and source links when an
unambiguous Safetensors checkpoint or selected GGUF file fits the inspection
budget. GGUF dimensions are labelled in their native GGML axis order, not silently
converted to PyTorch shapes. Multiple GGUF variants require an explicit file
selection. No graph is drawn.

**Checkpoint elements are not unique model parameters.** Adapter tensors,
quantization scales and repeated storage may be included. The parameter total
and architecture coverage therefore remain unknown. File presence also does not
mean every checkpoint header was inspected. JSON, Markdown, audit, diagnostics
and ZIP exports remain available; Mermaid and architecture questions do not.

To retest an old failed submission after an engine fix, submit its model ID
again. Existing saved failures are not rewritten into new results. The local
testing launcher increases the allowance without deleting quota history.

## Report Checks

| Check | What it measures |
| --- | --- |
| Semantic coverage | Classified components in the generated graph. |
| Module semantic coverage | Classified native modules, including the raw hierarchy. |
| Checkpoint parameter coverage | Unique native parameters matched to checkpoint metadata, weighted by parameter count. |
| Tensor shape coverage | Graph connection records with an established shape, not runtime-tested shapes. |
| Unknown nodes | Graph components whose semantic role is not established. |
| Unmapped checkpoint tensors | Checkpoint tensor records without a native module owner. |
| Forward executed | Whether model forward execution occurred; the current engine records No. |
| Runtime trace available | Whether runtime tracing supports this report; the current engine records No. |

Unknown means a check was not recorded reliably. It must not be presented as a
passing check. A collapsed stack may represent multiple structurally matching
layers; its evidence and source paths preserve those instances. Node parameter
counts overlap across parent/child views: do not add them together to estimate
the model's unique total.

## Warnings Versus Analysis Scope

**Warnings are specific extraction problems for this model.** They remain visible
and can make a report partial even when all coverage percentages are 100%.

**Analysis scope describes the method's boundaries**, not eight failures:

1. Static Python analysis supports a bounded subset; it cannot prove every branch.
2. Reconstructed relationships and symbolic dimensions are not runtime observations.
3. Edge dtypes describe native meta-model types; checkpoint dtypes come from headers.
4. Secondary outputs and opaque operations need established roles before shapes can be asserted.
5. Component containment is not automatically an execution connection.
6. Weight downloads, forward execution and remote model code are disabled by design.
7. Quantized/opaque/native-dependent variants may need separate supported paths.
8. Collapsed child graphs show representative instances, with original paths retained.

These boundaries remain visible; separating them from warnings does not remove
uncertainty or certify unsupported variants. For ViT, the accepted path is fixed
resolution, unmasked input, without positional interpolation. No image upload or
image-classification inference is performed. The generic Vocabulary field has no
text-vocabulary meaning for this image classifier; modality-specific summaries
remain a presentation gap.

## Check the Current ViT Report Yourself

1. Open your existing analysis. Refresh the page once if it still shows the old
   warning grouping; no new job or quota spend is needed for this presentation fix.
2. Confirm **Completed** near the model title and **Model revision** below the metrics.
3. Expand **Report checks**: all four coverage values should be 100%, unknown nodes
   zero, unmapped tensors zero, and forward/trace No.
4. Expand **Analysis scope**. The general method notes belong here. There should
   be no extraction warning list for this completed ViT report.
5. Inspect **Select tokens**: `[B,197,768]` becomes `[B,768]`. Inspect **Classifier**:
   input `[B,768]`, output `[B,1000]`, and only `classifier.bias`/`classifier.weight`
   in its tensor list.
6. Open **VIT Backbone → Embeddings → Patch Embedding**. The image tensor becomes
   196 patch tokens; adding CLS produces 197 tokens.
7. Export Architecture JSON. Compare `metadata.analysis_report` with Report checks
   and `metadata.limitations` with Analysis scope. The export remains the original
   stored report; only the API/UI presentation of legacy summaries is normalized.
   Legacy completion badges are also reassessed from that report so they cannot
   contradict its checks. The original database status is not rewritten.

## Current Closure Gate

No new model family, feature, AI rollout or deployment expansion is required to
close this report-clarity pass. Before claiming the existing product gap-free:

- Keep genuine engine warnings visible and fix them with source-backed tests.
- Resolve GPT-2's current `is_torch_fx_proxy` warning in native padding diagnostics;
  it is not an unmapped-parameter or missing-shape issue. Do not delete it blindly.
- Replace ambiguous modality-irrelevant summary fields with evidence-backed applicability.
- Audit large/repeated/nested graphs without confusing representative structure
  or connection coverage with verification of all runtime paths.
- Continue current job/storage/worker reliability checks; cloud and inference
  verification remain separate unfinished release gates, not features to rush into.

Every subsequent progress report should state: what changed, actual verification,
remaining gaps/blockers, how you can test, and the PR. Test failures and unverified
claims must be stated plainly. These are closure criteria, not a claim they are
all already satisfied.
