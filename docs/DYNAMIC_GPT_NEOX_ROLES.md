# GPT-NeoX semantic role increment

**2026-10-23.** This is a bounded improvement to the partial GPT-NeoX graph,
not a complete-model claim. The current engine already extracted graphs for
Pythia, but left its repeated attention, MLP and decoder-layer modules as
unknown. The installed native `GPTNeoXAttention` has a `query_key_value`
linear projection from H to 3H and a `dense` H-to-H projection. Its
`GPTNeoXMLP` has matching expansion and contraction projections with an
activation. Those module facts support semantic roles without running a
forward pass or inferring tensor transformations from names alone.

The classifier now recognizes those exact projection and activation
topologies. Incompatible projection dimensions remain unknown. The existing
decoder-block rule then recognizes the layer from its classified attention,
feed-forward and normalization subtree plus the causal-LM context. The engine
cache identity is `2026.10.23.3`; IR schema remains `1.1`.

| Pinned case | Before semantic / module coverage | After | Unknown native modules | After graph result |
| --- | --- | --- | --- | --- |
| `EleutherAI/pythia-70m@a39f36b100fe8a5377810d56c3f4789b9c53ac42` | 0.88 / 0.775 | 1.0 / 1.0 | 18 → 0 | 4 inspectable graphs, still partial |
| `EleutherAI/pythia-160m@50f5173d932e8e61f858120bcb800b97af589f46` | 0.88 / 0.7632 | 1.0 / 1.0 | 36 → 0 | 4 inspectable graphs, still partial |

The 70m input was its saved current-engine pinned report. The 160m config and
Safetensors header were fetched at the exact revision, reading 20,944 header
bytes and zero weight-payload bytes. Before/after comparisons replayed those
verified tensor records through the same engine with only the classifier
changed. Full replay artifacts and the 160m fetch audit are local ignored
diagnostics under `.local/analysis/`.

Both reports retain 100% checkpoint parameter coverage and seven warnings.
The warnings concern unproven view, transpose and reshape contracts; opaque
QKV chunking, rotary application and attention dispatch; and historical
unmapped checkpoint tensors. Neither report qualifies as complete. Tensor
shape coverage changes from 0.6053 to 0.5641 because the newly classified
backbone is inlined into the overview, removing one duplicate known-shape
boundary edge, while the attention graph exposes additional unknown edges.
This is a graph-denominator change, not proof of lost tensor facts. The next
increment should tackle a source-backed tensor operation or attention-call
contract, then rerun these pins and complete-model controls.
