# Transformer runtime

`TransformerBlock` is the block execution boundary. It owns no tensor memory
policy and contains no CUDA or model-family branch. Its dependencies are the
abstract `Attention`, `Norm`, `MoELayer`, and `TensorBackend` contracts.

```text
hidden-state batch
      │
      ▼
input RMSNorm (when mapped)
      │
      ▼
Attention (Q/K/V, causal scaled softmax, RoPE/cache, output projection)
      │ attention residual (when input norm is mapped)
      │
      ▼
RMSNorm
      │
      ▼
MoELayer (batch route, schedule, grouped experts, weighted sum)
      │
      ▼
residual add with the attention residual branch
```

`InferenceContext` carries batch size, sequence position, hidden dimension, and
layer index. After successful execution it records per-token routing decisions,
assignment and unique-expert counts, payload bytes, selected backend, and timing
for routing, scheduling, expert execution, and expert combination. Advancing a
layer clears only layer-local routing and execution state.

The Phase 10 `TransformerLayer`/`MoELayer::execute` entry point remains for API
compatibility. It still represents the earlier identity-attention wrapper.
Production-oriented Phase 12 callers use `TransformerBlock` and
`MoELayer::executeExpertsBatch`; the one-token expert API delegates to the same
batched implementation.

`TransformerModelRuntime` now composes mapped blocks across all declared layers.
Token embeddings, dense layers, final normalization/output heads, tokenizer,
generation loop, sampling, and server APIs remain outside this foundation.
