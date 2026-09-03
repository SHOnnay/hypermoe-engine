# Attention component

`Attention` separates transformer orchestration from an execution backend. Its
input consists of a rank-2 hidden-state batch and four projection views: query,
key, value, and output. `AttentionResult` retains Q/K/V, score, probability,
context, and output tensors so correctness tests and profilers can inspect every
stage.

`CpuAttention` is the portable FP32 reference implementation. It computes Q/K/V
through `TensorBackend::matmul`, forms scaled dot-product scores, applies a
row-wise numerically stable softmax, aggregates the value matrix, and applies the
output projection. It validates device, dtype, contiguity, ownership lifetime,
and every projection dimension before execution.

The reference currently models one attention head and applies no causal mask.
Those semantics are explicit rather than inferred from Qwen metadata. A future
CUDA or model-aware attention implementation will implement the same interface;
the transformer block will not need CUDA-specific logic.
