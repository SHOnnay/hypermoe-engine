# Attention component

`Attention` separates transformer orchestration from an execution backend. Its
input consists of a rank-2 hidden-state batch and four projection views: query,
key, value, and output. `AttentionResult` retains Q/K/V, score, probability,
context, and output tensors so correctness tests and profilers can inspect every
stage.

`CpuAttention` is the portable FP32 reference implementation. It computes Q/K/V
through `TensorBackend::matmul`, forms scaled dot-product scores, applies a
row-wise numerically stable softmax, aggregates the value matrix, and applies the
output projection. Phase 13 extends it with causal masking, multiple query heads,
grouped key/value heads, RoPE, and optional per-layer KV-cache use. It validates
device, dtype, contiguity, ownership lifetime,
and every projection dimension before execution.

The two-argument configuration defaults preserve the Phase 12 one-head,
non-causal behavior. Model runtime supplies explicit head, causal, rotary, layer,
and sequence-position configuration. A future CUDA implementation will implement
the same interface; the transformer block needs no CUDA-specific logic.
