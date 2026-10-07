# GPTNeoForCausalLM Hierarchy Preview

Model: `EleutherAI/gpt-neo-125m` | Revision: `21def0189f5705e2521767faed922f1f15e7d7db`

These graphs are generated from the same Hierarchical Model IR. Solid edges represent static source or pattern dataflow; dotted edges represent structural relationships. No forward pass was executed.

Child graphs of repeated nodes show a representative instance. All instance paths and parameters remain in the raw structure. Unknown shapes and components are explicitly marked.

## overview

```mermaid
flowchart TD
    n0["Token IDs"]
    n1["Position IDs"]
    n2["Token Embedding"]
    n3["Position Embedding"]
    n4["Add embeddings"]
    n5["Dropout"]
    n6["Decoder Block x12"]
    n7["Final LayerNorm"]
    n8["LM Head"]
    n9["Output"]
    n8 -->|"[B, S, 50257]"| n9
    n0 -->|"[B, S]"| n2
    n1 -->|"[1, S]"| n3
    n2 -->|"[B, S, 768]"| n4
    n3 -->|"[1, S, 768]"| n4
    n4 -->|"[B, S, 768]"| n5
    n5 -->|"[B, S, 768]"| n6
    n6 -->|"[B, S, 768]"| n7
    n7 -->|"[B, S, 768]"| n8
```

## transformer_h_0

```mermaid
flowchart TD
    n0["Hidden States"]
    n1["LayerNorm"]
    n2["Self Attention"]
    n3["Residual Add"]
    n4["LayerNorm"]
    n5["Feed Forward / MLP"]
    n6["Residual Add"]
    n7["Output"]
    n0 -->|"[B, S, 768]"| n1
    n1 -->|"[B, S, 768]"| n2
    n2 -->|"[B, S, 768]"| n3
    n0 -->|"skip: [B, S, 768]"| n3
    n3 -->|"[B, S, 768]"| n4
    n4 -->|"[B, S, 768]"| n5
    n3 -->|"skip: [B, S, 768]"| n6
    n5 -->|"[B, S, 768]"| n6
    n6 -->|"[B, S, 768]"| n7
    n2 -->|"[unknown shape]"| n7
```

## transformer_h_0_attn

```mermaid
flowchart TD
    n0["Hidden States"]
    n1["Self Attention"]
    n2["Output"]
    n0 -->|"[B, S, 768]"| n1
    n1 -->|"[B, S, 768]"| n2
```

## transformer_h_0_attn_attention

```mermaid
flowchart TD
    n0["Hidden States"]
    n1["Projection"]
    n2["Projection"]
    n3["Projection"]
    n4["Matrix multiply"]
    n5["Dropout"]
    n6["Matrix multiply"]
    n7["Projection"]
    n8["Dropout"]
    n9["Output"]
    n0 -->|"[B, S, 768]"| n1
    n0 -->|"[B, S, 768]"| n2
    n0 -->|"[B, S, 768]"| n3
    n1 -->|"[B, S, 768]"| n4
    n2 -->|"[B, S, 768]"| n4
    n4 -->|"[unknown shape]"| n5
    n5 -->|"[unknown shape]"| n6
    n3 -->|"[B, S, 768]"| n6
    n6 -->|"[unknown shape]"| n7
    n7 -->|"[B, S, 768]"| n8
    n8 -->|"[B, S, 768]"| n9
    n5 -->|"[unknown shape]"| n9
```

## transformer_h_0_mlp

```mermaid
flowchart TD
    n0["Hidden States"]
    n1["Projection"]
    n2["Activation"]
    n3["Projection"]
    n4["Dropout"]
    n5["Output"]
    n0 -->|"[B, S, 768]"| n1
    n1 -->|"[B, S, 3072]"| n2
    n2 -->|"[B, S, 3072]"| n3
    n3 -->|"[B, S, 768]"| n4
    n4 -->|"[B, S, 768]"| n5
```
