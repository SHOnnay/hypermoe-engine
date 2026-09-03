# Transformer runtime

`TransformerBlock` is the Phase 12 execution boundary. It owns no tensor memory
policy and contains no CUDA or model-family branch. Its dependencies are the
abstract `Attention`, `Norm`, `MoELayer`, and `TensorBackend` contracts.

```text
hidden-state batch
      │
      ▼
Attention (Q/K/V, scaled softmax, context, output projection)
      │
      ▼
RMSNorm
      │
      ▼
MoELayer (batch route, schedule, grouped experts, weighted sum)
      │
      ▼
residual add with attention output
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

This is one transformer block, not complete inference. Multi-head and causal
attention, positional encoding, KV caching, dense layers, tokenizer, generation
loop, sampling, and server APIs are intentionally outside Phase 12.
