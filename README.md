# Model Anatomy

> **Reconstruct an open-weight neural network into a semantic, tensor-aware architecture graph — then let us inspect what the model actually contains.**

Model Anatomy is an exploration and reconstruction tool for Hugging Face models.

It starts from the artifacts around an open-weight model — configuration, executable `nn.Module` structure, graph information, source code, and Safetensors metadata — and combines them into one representation that a human can actually understand.

## Why

A model repository gives us pieces of the truth. The engineer still has to mentally reconstruct:

- how `nn.Module` parts connect
- what those modules mean architecturally
- which tensors belong to which component
- how dimensions flow
- where repeated blocks begin/end
- what the actual computation structure looks like

Model Anatomy turns that investigation into a generated model graph.

## Core pipeline

```text
Hugging Face Model
        ↓
Artifact Resolver
        ↓
Config + nn.Module + FX/Graph + Source + Safetensors
        ↓
Raw Architecture IR
        ↓
Semantic Architecture IR
        ↓
Interactive Model Graph
        ↓
Tensor / Parameter / Evidence Inspection
        ↓
Model Analysis / Autopsy
```

## The key distinction

This is **not just a diagram generator**.

The diagram is the visible result of an underlying reconstruction engine.

The long-term question is:

> Can we automatically reconstruct enough of a real neural network from its open-weight artifacts to make its architecture, tensors, and relationships directly inspectable?

## First milestone

Start with **one decoder-only Transformer**.

The first implementation must reliably produce:

- module hierarchy
- semantic components
- tensor names
- tensor shapes and dtype
- parameter counts
- graph relationships where traceable
- evidence for where each fact came from

## Important technical boundary

Safetensors gives valuable tensor-level information, but it does **not** by itself contain the complete computation graph.

Therefore we fuse multiple sources:

- `config.json` / `AutoConfig`
- instantiated `nn.Module`
- `named_modules()`
- graph tracing such as `torch.fx` where possible
- source code
- Safetensors metadata

## Product layers

### Layer 1 — Reconstruction
Turn model artifacts into a canonical representation.

### Layer 2 — Visualization
Generate an architecture diagram from that representation.

### Layer 3 — Inspection
Expose tensors, parameters, dimensions, implementation and evidence.

### Layer 4 — Analysis
Eventually support Model Autopsy: comparisons, anomalies, contradictions and deeper findings.

## Current status

**Ground established. Implementation starts with reconstruction.**

See [`docs/`](docs/) for the project context and decisions.
