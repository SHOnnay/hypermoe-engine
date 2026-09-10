# GPU dataflow

Phase 18 reduces host synchronization while preserving the CPU reference path.
`CudaTensorBackend` uses cuBLAS for FP32 matrix multiplication and optional CUDA
kernels for activation, RMSNorm, RoPE, router softmax/top-k, causal attention,
and grouped expert gather/scatter. Backend selection remains explicit; no code
assumes a particular NVIDIA product.

## Resident execution path

Router logits remain on the device through probability normalization and top-k
selection. Only the compact expert IDs and routing weights cross to the host,
because the scheduler still makes host-side residency decisions. Hidden states
are gathered on-device for each selected expert, expert projections execute with
cuBLAS, and weighted outputs are scattered back with atomic accumulation.

Attention projects Q/K/V with cuBLAS, applies RoPE on the device, and appends to
a geometrically growing contiguous `CudaKVCache`. The attention kernel consumes
a logical device snapshot and performs causal masking, stable softmax, and
context accumulation without materializing full score or context tensors in
host RAM. The output projection remains cuBLAS-backed.

## Synchronization contract

Tensor operations enqueue work on the compute stream. Public materialization,
cross-stream copies, scheduler-visible routing results, and explicit
`synchronize()` calls are synchronization boundaries. Backend statistics count
those boundaries, and the GPU benchmark reports measured transfer, resident
compute, and synchronization metrics. GPU-utilization fields remain `null` until
a hardware counter provider is connected.

## Fallbacks and limitations

If CUDA is absent, the CPU implementation is complete. If runtime/cuBLAS are
available but no CUDA compiler is configured, HyperMoE retains the Phase 17
staged reference operations. The initial kernels prioritize deterministic FP32
correctness; they are not yet fused, warp-specialized, quantized, or paged.
