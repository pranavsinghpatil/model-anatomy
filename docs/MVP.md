# MVP

## Promise
Paste a Hugging Face model and get a generated technical architecture representation that can be inspected down to modules and tensors.

## Input
- Hugging Face model ID
- Hugging Face model URL

## Reconstruction
- model/config resolution
- one decoder-only Transformer family
- module hierarchy
- tensor metadata
- initial graph relationships where available
- Raw IR
- Semantic IR

## Explorer
- architecture graph
- repeated blocks
- node details
- tensors
- dimensions
- parameter counts
- evidence/source details

## Later
- reference comparison
- checkpoint diffs
- anomaly detection
- provenance/autopsy
- broader architecture support

## Success
A technically strong user understands the model faster than from its raw module tree.
