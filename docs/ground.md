# Project Ground

## Product

Model Anatomy reconstructs a Hugging Face checkpoint into a semantic, inspectable model graph.

## Inputs

- model ID / URL
- config / AutoConfig
- instantiated `nn.Module`
- module tree / `named_modules()`
- graph representation such as `torch.fx` where traceable
- model source code
- Safetensors metadata

## Outputs

- Raw Architecture IR
- Semantic Architecture IR
- architecture graph
- tensor map
- parameter map
- evidence links

## Later layer: Model Autopsy

Use the reconstructed representation to investigate:

- checkpoint changes
- modification boundaries
- artifact contradictions
- construction clues
- weight anomalies
- localized provenance

## First build

`HF model → resolve → inspect → reconstruct → emit IR → render one trustworthy graph`
----
# Model IR

The Model IR is the internal representation between raw artifacts and the frontend.

## Design principle

Do not force every model into a Transformer-shaped ontology.

The IR should represent what exists.

## Initial conceptual structure

```python
Model
├── metadata
├── architecture
│   ├── components
│   ├── hierarchy
│   └── relationships
├── tensors
├── parameters
├── operations
└── evidence
```

## Component

A component may represent:

- embedding
- attention
- normalization
- MLP
- convolution
- encoder block
- decoder block
- text encoder
- denoiser
- VAE
- projection head
- custom module

## Tensor

A tensor record should eventually include:

- name
- shape
- dtype
- parameter ownership
- source artifact
- semantic association where established

## Relationship

Relationships represent things such as:

- contains
- calls
- feeds
- residual connection
- owns parameter
- produces tensor

## Evidence

Facts should be traceable to an artifact or transformation.

Example:

```text
Fact:
  q_proj is a Linear layer

Evidence:
  nn.Module inspection
  module path: encoder.layers.0.attention.q_proj

Confidence:
  direct
```

Derived facts should say that they are derived.
----
# Evidence Model

The system must distinguish what it **observed** from what it **inferred**.

## Evidence classes

### Direct
Read directly from an artifact.

Example: `q_proj.weight` has shape `[4096, 4096]`.

### Derived
Calculated from direct facts.

Example: parameter count = product of tensor dimensions.

### Reconstructed
Obtained by combining multiple artifacts.

Example: mapping a tensor to a semantic attention projection by combining module structure and tensor ownership.

### Unknown
The available artifacts do not establish the fact reliably.

## Why this matters

The system should never manufacture certainty merely because an architecture pattern is familiar.

A missing connection should be allowed to remain:

```text
UNKNOWN / NOT TRACEABLE
```

rather than becoming a plausible-looking hallucination.

## Future Autopsy rule

Any forensic finding should eventually follow:

```text
Observation
    ↓
Evidence
    ↓
Comparison / transformation
    ↓
Finding
    ↓
Confidence
```
-----
# Roadmap

## Phase 0 — Ground

- project thesis fixed
- scope fixed
- first architecture family selected

## Phase 1 — One real model

Build:

```text
HF model
→ resolve
→ instantiate
→ inspect modules
→ inspect tensors
→ Raw IR
```

Success condition:

**We can inspect one real model programmatically.**

## Phase 2 — Graph reconstruction

Add:

- tracing / FX where possible
- module relationships
- tensor-to-module mapping
- semantic component extraction

Success condition:

**We can generate a trustworthy architecture graph.**

## Phase 3 — Explorer

Add:

- graph UI
- node selection
- tensor details
- dimensions
- parameter counts
- evidence panel

Success condition:

**A developer can understand the model faster than by reading the raw tree.**

## Phase 4 — Model diversity

Test different architecture families.

Start with a second family only after the first pipeline works.

## Phase 5 — Autopsy

Only then explore:

- reference comparison
- checkpoint differences
- unusual structures
- contradictions
- forensic findings

## Phase 6 — Polish

- demo flow
- performance
- failure states
- documentation
- reproducibility
-----
# Decisions

## D001 — Working name

**Model Anatomy**

The name may change later. The technical direction does not depend on it.

## D002 — Diagram is not the whole product

The diagram is the visible output of a reconstruction engine.

## D003 — Reconstruction before Autopsy

Autopsy depends on a trustworthy model representation.

Therefore:

```text
reconstruct first
analyze second
```

## D004 — Safetensors is not the graph

Safetensors is a tensor evidence source.

We must combine it with executable structure and other artifacts.

## D005 — Do not force one architecture template

The representation must support model-specific structures.

## D006 — Evidence is first-class

The system should be able to answer:

> “Why do you think this is here?”

## D007 — One model before many

A reliable end-to-end pipeline for one model is more valuable than shallow support for ten families.
----
# Demo Story

## Opening

Paste a Hugging Face model.

## Step 1 — Resolve

Show that the system discovers the model artifacts.

## Step 2 — Reconstruct

Show:

- config
- module hierarchy
- tensors
- graph information

## Step 3 — Generate

Produce a clean architecture diagram automatically.

## Step 4 — Inspect

Click a component.

Show:

- semantic role
- implementation path
- tensors
- shapes
- dtype
- parameter count
- evidence

## Step 5 — Surprise

Show one piece of information that would normally require manual investigation.

This is critical.

The demo should not end with:

> “Look, we made a pretty graph.”

It should end with:

> “The graph lets us inspect the actual model.”

## Later

Use the same representation for Model Autopsy.

The story becomes:

```text
Understand the model
        ↓
Inspect the model
        ↓
Investigate the model
```
----
# Open Questions

These are questions to answer by implementation, not endless discussion.

1. Which model should be the first target?
2. How much can `torch.fx` reconstruct reliably for that model?
3. How should tensors map to semantic components?
4. What is the smallest useful IR?
5. How should repeated blocks be represented?
6. How do we handle unsupported/custom code?
7. How do we communicate uncertain graph edges?
8. What makes the generated diagram genuinely better than an `nn.Module` printout?
9. What is the first insight we can expose that HF does not present directly?
10. Which second architecture family proves that the IR is not hard-coded?

## Immediate question

**Pick one real HF model and get from URL → reconstructed graph.**

That is the next engineering milestone.
----
