# GPT2LMHeadModel Hierarchy Preview

Model: `openai-community/gpt2` | Revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`

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
    n6["Transformer Block x12"]
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
    n1 -->|"[1, S]"| n6
    n6 -->|"[B, S, 768]"| n7
    n7 -->|"[B, S, 768]"| n8
```

## transformer_block

```mermaid
flowchart TD
    n0["Hidden States"]
    n1["LayerNorm"]
    n2["Multi-Head Attention"]
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
```

## attention

```mermaid
flowchart TD
    n0["Input"]
    n1["QKV Projection"]
    n2["Attention Core"]
    n3["Output Projection"]
    n4["Dropout"]
    n5["Output"]
    n0 -->|"[B, S, 768]"| n1
    n1 -->|"[B, S, 2304]"| n2
    n2 -->|"[B, S, 768]"| n3
    n3 -->|"[B, S, 768]"| n4
    n4 -->|"[B, S, 768]"| n5
```

## attention_core

```mermaid
flowchart TD
    n0["Packed QKV"]
    n1["Split QKV and reshape heads"]
    n2["Q"]
    n3["K"]
    n4["V"]
    n5["Q @ K transpose"]
    n6["Scale attention scores"]
    n7["Attention mask / position bias"]
    n8["Softmax"]
    n9["Dropout"]
    n10["Attention probabilities @ V"]
    n11["Transpose sequence and head axes"]
    n12["Merge heads"]
    n13["Context"]
    n0 -->|"[B, S, 2304]"| n1
    n1 -->|"[B, 12, S, 64]"| n2
    n1 -->|"[B, 12, S, 64]"| n3
    n1 -->|"[B, 12, S, 64]"| n4
    n2 -->|"[B, 12, S, 64]"| n5
    n3 -->|"[B, 12, S, 64]"| n5
    n5 -->|"[B, 12, S, S]"| n6
    n6 -->|"[B, 12, S, S]"| n7
    n7 -->|"[B, 12, S, S]"| n8
    n8 -->|"[B, 12, S, S]"| n9
    n9 -->|"[B, 12, S, S]"| n10
    n4 -->|"[B, 12, S, 64]"| n10
    n10 -->|"[B, 12, S, 64]"| n11
    n11 -->|"[B, S, 12, 64]"| n12
    n12 -->|"[B, S, 768]"| n13
```

## feed_forward

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
