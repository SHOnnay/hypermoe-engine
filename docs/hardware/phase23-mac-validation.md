# Phase 23 Mac validation and RTX handoff

Baseline synchronized with origin/main: `14ef959`. Previous INT8, overlap,
adaptive-residency and Hermes changes were preserved. Environment: Darwin
arm64, AppleClang 21.0.0.21000334, strict C++20 with warnings as errors.

## Implementation

- Cooperative affine INT8 GEMM: 32 columns x 8 K lanes, shared input tiles,
  bounded split-K and deterministic same-stream FP32 partial reduction.
- Reference/auto/cooperative dispatch, with no artifact-format changes.
- Deferred event completion/timing with retained tensor/scratch ownership;
  removed GEMM/elementwise/activation profiler-only waits.
- Tensor-copy compute-to-transfer dependencies and one required host completion
  event wait; no redundant post-copy stream wait.
- Explicit CUDA leaf/region, transfer and host-wait profiling alongside legacy
  aggregate metrics; no synthetic GPU utilization.

The memory hierarchy, transfer readiness/leases, LRU/Hybrid residency,
prediction, INT8 projection layout, wide attention, GQA, Q/K normalization and
RoPE algorithms are unchanged. The attention change is an inclusive timing
scope, not an attention optimization.

## Validation

Release and Debug complete builds pass. Each full CTest suite passes 26/26,
including Phase 20, 21, 22A, 22B and 22C regressions. Debug with CUDA requested
detects the absent toolkit and falls back cleanly. AddressSanitizer and UBSan
also pass 26/26 with halt-on-error enabled.

An initial `ASAN_OPTIONS=detect_leaks=1` run aborted all processes before test
execution: this Apple ASan reports leak detection unsupported. The successful
run used `detect_leaks=0`; leak detection is **not** claimed. Device kernels
are not instrumented by these Mac sanitizers.

Phase 23 exercises CPU launch validation, host emulation of cooperative
partition/tile/reduction order, event ownership with/without profiling,
asynchronous completion, error propagation, exception/shutdown barriers,
legacy metrics and report compatibility, CPU wide attention/GQA/QK norm/RoPE,
and CPU INT8 expert MLP. Conditional native tests additionally compare eight
projection shapes, CPU/scalar/cooperative outputs, an independent affine
oracle, released tensor owners, complete INT8 MLP, wide attention intermediates,
and absence of profiler waits. Native checks are explicitly skipped on Mac.

The CUDA-enabled C++ tensor-backend branch received a syntax-only check with
temporary API declarations. This catches C++ branch errors but is **not** a
CUDA toolkit build, ABI validation, NVCC compilation, or device execution.
Native compilation and numerical/device-memory qualification remain pending.

## Benchmarks

`hypermoe_int8_gemm_benchmark` ran and emitted:

```json
{"cuda_available":false,"gpu_utilization_percent":null,"measurements":[]}
```

No native CUDA or Qwen3-30B-A3B checkpoint is available here. Phase 23 RTX
throughput, TTFT, expert GEMM/attention latency, utilization, and VRAM results
are **not measured**. The historical 0.158 tokens/sec remains Hermes' Phase 22C
baseline, not evidence of a Phase 23 improvement.

The unchanged deterministic CPU expert fixture (hidden 128, intermediate 256,
40 repetitions) ran: FP32 first-expert latency 0.104792 ms; INT8 0.101041 ms.
Its measured steady-state rates are 7701.439823 and 5166.346009 expert
executions/sec respectively, **not** Qwen or generated-text throughput.
INT8-vs-FP32 mean/max output error: 0.000407 / 0.000919. Synthetic cache/transfer
figures from this fixture are not hardware transfer measurements or Phase 23
optimization results.

## Files changed

```text
CMakeLists.txt
README.md
benchmarks/cuda_compute/int8_gemm_benchmark.cpp
benchmarks/profile_real_model/profile_real_model_benchmark.cpp
docs/architecture.md
docs/design-decisions.md
docs/components/gpu-compute.md
docs/hardware/phase23-mac-validation.md
src/backend/Backend.hpp
src/backend/cuda/CudaKernels.cu
src/backend/cuda/CudaKernels.hpp
src/backend/cuda/CudaMemoryManager.cpp
src/backend/cuda/CudaRuntime.cpp
src/backend/cuda/CudaRuntime.hpp
src/backend/cuda/Int8GemmPlan.cpp
src/backend/cuda/Int8GemmPlan.hpp
src/experts/ExpertExecutor.cpp
src/models/runtime/PackedModelRuntime.cpp
src/models/runtime/PackedModelRuntime.hpp
src/profiling/GpuEventQueue.cpp
src/profiling/GpuEventQueue.hpp
src/profiling/GpuOperation.hpp
src/profiling/Profiler.cpp
src/profiling/Profiler.hpp
src/profiling/RealModelProfile.cpp
src/profiling/RealModelProfile.hpp
src/tensor/activation/Activation.cpp
src/tensor/backend/CudaTensorBackend.cpp
src/tensor/backend/CudaTensorBackend.hpp
src/transformer/attention/CudaAttention.cpp
tests/phase23_tests.cpp
```

## Remaining qualification

Follow the [compute guide](../components/gpu-compute.md) on RTX 4070/CUDA 12.5:
native Release/Debug, all tests without native skips, Compute Sanitizer, scalar
vs cooperative microbenchmarks with profiling on/off, and identical-artifact
Qwen CPU/CUDA trace and real-model comparisons at fixed residency budgets.
Keep CUDA 12.5 `bin` on Windows PATH for DLL discovery.

Serial reference attention/RMSNorm/router, router readback, gather/scatter
waits, FP32 static GEMM, event/launch overhead and residency misses remain
candidates. CUDA event spans are not a full Nsight/CUPTI timeline; implicit
allocation/pageable-copy waits are outside the explicit wait counter. The next
phase should target the measured dominant stage after this hardware gate.
