# Real-model profiling

`hypermoe_profile_real_model` executes caller-provided token IDs sequentially
through one packed runtime and bounded KV cache. The JSON report contains only
values measured or accounted during that run.

## Metrics

- model identity, physical parameter count, layer count, configured experts,
  layer-qualified active experts, and token count;
- construction time, first-token latency, average subsequent decode latency,
  total throughput, and tokens per second;
- persistent shared-tensor storage/execution bytes, current expert residency,
  logical KV-cache bytes, and runtime-accounted RAM/VRAM;
- expert requests, cache hits/misses, expert bytes loaded from storage,
  per-layer expert frequency, and prefetch request/hit/miss accuracy.

CPU memory reports place CPU-backed expert device buffers and static tensors in
RAM. CUDA reports place them in VRAM. These are runtime ownership/accounting
figures, not process RSS or `nvidia-smi` peaks; allocator reservations, driver
context memory, and short-lived intermediate tensors require an external
hardware sampler for a complete peak measurement.

## Correctness workflow

`hypermoe_real_qwen_validate` first completes and captures the CPU reference. If
CUDA is available, it constructs the same packed graph on CUDA and compares
embedding output, every attention output, every selected expert output, every
layer output, final normalization, and logits at the FP32 oracle tolerance. A
CPU-only machine emits an explicit skipped CUDA result instead of zeros described
as GPU measurements.

Benchmark reports are meaningful only when they record the artifact identity,
token IDs, build mode, hardware, driver, and CUDA versions externally. Phase 20
does not bundle or claim results for a production Qwen checkpoint.
