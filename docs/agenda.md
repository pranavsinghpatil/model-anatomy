# Project Ground

This is the source of truth for what we are building right now.

## One sentence

**Model Anatomy reconstructs an actual open-weight neural network into a semantic, tensor-aware architecture graph that a developer can inspect instead of manually reverse-engineering the model.**

## The user problem

Hugging Face can expose model cards, configs, source code and weights, but understanding a particular model can still require manual investigation.

The painful part is connecting the pieces:

```text
nn.Module hierarchy
      +
tensor names / shapes
      +
implementation
      +
configuration
      +
actual computation
      ↓
human mental model
```

We want the system to perform that reconstruction.

## What the user gets

A model-specific architecture diagram similar in spirit to a paper architecture figure, but generated from the actual model artifacts.

The graph should be:

- structural
- tensor-aware
- connected
- semantic
- inspectable
- evidence-backed

## What makes it different

We are not asking an LLM to invent an architecture explanation.

We are reconstructing it from artifacts and representing what we can establish.

## Project layers

```text
RECONSTRUCTION
    ↓
SEMANTIC MODEL REPRESENTATION
    ↓
VISUAL EXPLORATION
    ↓
DEEPER ANALYSIS
```

Model Autopsy belongs above reconstruction. It is not the first thing we build.

## Current scope

One real model first.

Prove the pipeline.

Then expand architecture families.

## Hard constraint

Every impressive frontend feature is secondary until the backend can produce a trustworthy representation of an actual model.

----
# Problem

## The real pain

A developer finds an open-weight model and wants to understand its architecture.

They may encounter:

- a large `nn.Module` tree
- hundreds or thousands of tensor names
- configuration fields
- implementation-specific class names
- repeated blocks
- tensor dimensions scattered across weights
- custom code
- model-specific conventions

The information exists, but the **relationship between the pieces is expensive to reconstruct manually**.

## Target question

> What would an experienced engineer normally have to inspect manually before they could draw a trustworthy architecture diagram of this exact checkpoint?

That is the work Model Anatomy tries to automate.

## Not the problem

We are not primarily solving:

- model search
- model-card summarization
- generic chat with a model
- static diagram drawing
- benchmark aggregation

## Product thesis

**The useful unit is not the model file. It is the reconstructed model representation.**

Once we have that representation, visualization and analysis become possible.
----
# Goals

## Primary goal

Automatically reconstruct a real Hugging Face model into a representation that exposes:

1. architecture
2. module hierarchy
3. computation relationships where traceable
4. tensor mapping
5. tensor dimensions
6. parameters
7. semantic roles
8. evidence

## Demo goal

A judge should be able to:

1. paste a model
2. see the system inspect it
3. see a generated architecture
4. click a component
5. inspect its tensors and dimensions
6. inspect why the system believes the component exists
7. understand something they would otherwise have had to investigate manually

## Technical goal

Build an architecture reconstruction pipeline rather than a collection of UI tricks.

## Longer-term goal

Use the reconstructed representation for:

- model comparison
- checkpoint diffing
- anomaly detection
- implementation/metadata contradiction detection
- deeper Model Autopsy
- runtime/hardware analysis

These are extensions, not MVP requirements.
----
# Non-Goals

For the first version we explicitly do **not** build:

- a generic chatbot
- a model marketplace
- a model search engine
- a model-card summarizer
- a static diagram editor
- a universal parser for every architecture
- a complete provenance platform
- a full benchmark platform
- a production hardware optimizer
- an LLM explanation layer that simply guesses what the model does

## Rule

If a feature does not strengthen **actual model reconstruction or inspection**, it should probably wait.
----
# System Architecture

## Pipeline

```text
                  Hugging Face Model
                         │
                         ▼
                  Artifact Resolver
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Config         nn.Module     Safetensors
          │              │              │
          │              ▼              │
          │          Module Tree        │
          │              │              │
          │              ▼              │
          │           FX / Graph        │
          └──────────────┼──────────────┘
                         ▼
                Raw Architecture IR
                         │
                         ▼
              Semantic Architecture IR
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         Architecture  Tensors   Parameters
              │          │          │
              └──────────┼──────────┘
                         ▼
                Interactive Explorer
                         │
                         ▼
                   Model Analysis
```

## Artifact roles

### Config
Useful for declared architecture-level facts.

### `nn.Module`
Provides executable implementation structure and actual module classes.

### Module tree
Provides hierarchy and parameter ownership.

### FX / graph tracing
Provides operation/edge information where the model is traceable.

### Source code
Useful for custom semantics and cases that tracing cannot establish cleanly.

### Safetensors
Provides tensor-level facts such as names, shapes and dtype.

## Important rule

No single artifact is treated as the whole truth.

The reconstruction engine combines evidence.
----
