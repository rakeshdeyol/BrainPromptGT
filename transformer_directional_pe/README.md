# Standalone Transformer with Directional Positional Encoding

This folder is independent of the existing BrainPrompt model.

It implements:

- directed incoming and outgoing random-walk positional encoding;
- multi-head self-attention from scratch using Q, K, V projections;
- pre-normalized Transformer blocks;
- graph-level mean pooling and classification.

Input tensors:

```text
node_features: [batch, nodes, input_features]
adjacency:     [batch, nodes, nodes]
logits:        [batch, classes]
```

Run the synthetic example from the repository root:

```powershell
.\brainprompt_env\Scripts\python.exe -m transformer_directional_pe.example
```
