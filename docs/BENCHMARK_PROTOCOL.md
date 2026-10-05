# Benchmark Protocol

The benchmark is part of the product, not an afterthought.

## Purpose
Measure whether Model Anatomy reconstructs model structure faithfully and makes technical understanding easier than raw implementation inspection.

## Dimensions

### Reconstruction
component coverage
hierarchy fidelity
repeated-block detection
graph-edge recovery
tensor-to-module mapping

### Accuracy
tensor shape correctness
dtype correctness
parameter-count correctness
semantic-label correctness

### Robustness
traceable model
partially traceable model
custom-code model
unsupported model

### Human utility
Test whether technical users can answer predefined architecture questions faster with Model Anatomy than with the raw repository/code.

## Benchmark manifest
Pin model IDs and revisions under benchmarks/. Each case records architecture family, expected structural facts, difficult cases, and resource notes.

## Rules
1. Pin model revisions.
2. Record tool commit and environment.
3. Never silently change expected answers.
4. Separate automated metrics from human evaluation.
5. Record failures as well as successes.
6. Run before milestone completion and final release.

## Required result
timestamp, commit SHA, environment, per-model results, aggregate results, failures, limitations.