# CUDA inference

Phase 17 connects HyperMoE's existing tensor, expert, transformer, cache, and
generation interfaces to CUDA without removing their CPU implementations. CUDA
remains optional: if CMake cannot find the CUDA Toolkit and cuBLAS, the same
library and tests build with the CPU backend.

## Execution boundary

The CUDA tensor backend currently executes these FP32 operations in VRAM:

- host/device and device/device tensor copies through `cudaMemcpyAsync`;
- matrix multiplication through `cublasSgemm`;
- residual/vector addition through `cublasScopy` plus `cublasSaxpy`;
- elementwise multiplication through `cublasSdgmm`;
- RMSNorm through `cublasSnrm2`, `cublasSdgmm`, and `cublasSscal`;
- expert gate, up, and down projections through the generic
  `ExpertMlpExecutor`;
- attention QKV and output projections through `CudaAttention`.

The following correctness paths intentionally stage through the CPU reference:
embedding lookup, SiLU/GELU, router scoring/top-k, RoPE, causal attention
score/softmax/context, grouped expert gather/scatter, and vocabulary-by-hidden
tied LM heads. Hidden states and component outputs remain device tensors. These
boundaries are functional integration seams, not claims of end-to-end GPU
acceleration.

## Backend selection

An `InferenceConfig` declares `Device::cpu()` or `Device::cuda(ordinal)`. The
configured `GenerationModel`, `KVCacheManager`, and session device must match.
For CUDA, construct the model graph with `CudaTensorBackend`, `CudaAttention`,
`CudaRouterBackend`, and CUDA-capable normalization/expert components. Construct
the cache manager with the same tensor backend. A mismatch fails before a cache
session is allocated.

The generation API is otherwise unchanged. CUDA logits are copied to a bounded
host tensor through `GenerationModel::materializeHost` before sampling; code
never dereferences a device pointer on the host.

## Memory flow

Expert storage still moves through the existing NVMe-to-host-to-device transfer
pipeline. Expert projection views reuse the resident device buffer. Intermediate
expert and transformer tensors come from `CudaMemoryPool`. `CudaKVCache` owns
key/value device tensors for every contiguous append and reports conservative
per-session memory through `KVCacheManager`. Its current snapshot copies data to
host because attention softmax is still the reference implementation.

## Validation and benchmark

`hypermoe_phase17_tests` cleanly skips CUDA assertions when no CUDA runtime is
available. On a CUDA host it compares GEMM, addition, multiplication, attention,
expert execution, normalization, KV snapshots, and session device selection
against CPU values at the FP32 oracle tolerance (`1e-5`). Earlier CPU and CUDA
tests remain enabled.

`hypermoe_gpu_inference_benchmark` writes `hypermoe.gpu-inference.v1` JSON. It
always measures the CPU expert reference. CUDA initialization, H2D transfer,
expert latency, attention latency, approximate layer throughput, and allocation
fields are numbers only when a CUDA device is available; otherwise they are
`null`. This prevents a CPU development machine from fabricating RTX results.

## Current limitations

- CUDA execution is FP32 only.
- Attention score, softmax, context, and cache snapshots are not GPU-native.
- Activation, routing, embedding, tied-head projection, and MoE
  gather/scatter incur synchronization and PCIe transfers.
- KV storage is chunked rather than paged and does not reclaim individual pages.
- The repository contains no production tokenizer artifact parser, server, CUDA
  graph capture, fused kernel, or quantized CUDA GEMM.
- RTX 4070 performance for this Phase 17 benchmark must be collected on the
  qualified Windows host; existing hardware qualification numbers are not
  relabeled as Phase 17 inference results.
