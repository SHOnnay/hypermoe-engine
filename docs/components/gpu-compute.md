# Phase 23: CUDA compute foundation

## Audit and scope

Baseline: `14ef959`, preserving `62f7d17` INT8 packing/execution,
`26266ea` transfer overlap, and `0968bba` adaptive residency. Hermes reports
0.158 tokens/sec for the Qwen3-30B-A3B budget sweep. That report has no kernel
timeline: it does **not** establish which kernels dominate, SM utilization, or
whether execution is now compute-bound. Treat the throughput as a separately
measured baseline, not a Phase 23 measurement.

Source inspection found these candidates:

| Operation | Existing implementation | Limitation / Phase 23 action |
| --- | --- | --- |
| INT8 expert GEMM | One thread per output, serial FP32 accumulation over K | Only 3 blocks for `[1,2048] x [2048,768]`; cooperative split-K added |
| FP32 GEMM | cuBLAS SGEMM | Keep library implementation and FP32 precision contract |
| Attention scores/softmax/context | One thread per query/head, serial reductions | Still reference implementation; instrument, do not replace |
| RMSNorm/router | Serial reduction per row/token | Still reference implementation; instrument |
| Activations/RoPE | Parallel elementwise CUDA kernels | Keep math; remove activation's profiler-only wait |
| Profiling | GEMM/elementwise/activation completion waits | Replace with deferred event queries |
| Tensor copies | Host compute wait, copy event wait, then stream wait | Stream dependency plus one required host completion wait |

No Tensor Core path is claimed. INT8 storage with FP32 activations and
affine per-projection scales is not directly an INT8-by-INT8 Tensor Core GEMM.
Changing activation precision would change the numerical contract and is outside
this phase. Increasing independent CUDA blocks is the highest-confidence source
candidate, not a demonstrated speedup; validate before choosing later work.

## Cooperative INT8 GEMM

`Int8GemmPlan` chooses a reference or cooperative launch. Auto selects the
cooperative path for K >= 256 and N >= 64; small fixtures retain the scalar path.
Explicit `reference` and `cooperative` modes support reproducible comparison.

Each block has 32 column lanes and 8 K lanes (256 threads). Neighboring lanes
read neighboring signed INT8 weights; a 256-element FP32 input tile is shared
across all columns. The 2 KiB shared allocation holds inputs and reduction sums.
K is split into up to eight balanced partitions, followed by a fixed-order FP32
reduction kernel on the **same compute stream**. There are no atomic sums.
Odd widths, odd K, incomplete tiles, and single-partition cases are supported.

The Qwen gate/up mapping creates 96 blocks instead of 3, with 12 KiB of
temporary FP32 partials. The down mapping `[1,768] x [768,2048]` creates 128
blocks with 16 KiB of partials. These are launch counts, not measured occupancy.
Scratch comes from the existing CUDA memory pool; weights remain compressed
through every tier. No model format, scale/zero-point, expert offset, residency
policy, attention layout, or transfer subsystem changes.

Reduction order changes, so bitwise equality is not promised. Correctness tests
use `abs(error) <= 1e-5 + 1e-5 * abs(reference)` against the same affine INT8
representation, not against an unquantized model. Tests additionally compare
the retained scalar CUDA path and an independent double-accumulation oracle.

## Stream ownership, completion, and lifetimes

The scheduler/TransferManager continues to own expert transfer readiness.
The tensor backend owns its compute stream and separate tensor-copy transfer
stream. Expert A/B overlap, pinned staging, and expert leases are unchanged.

For tensor copies, an event after prior compute is waited on by the transfer
stream, replacing a host compute-stream wait. The copy's completion event is
still host-waited: `copyTensor` promises completion and must protect caller host
storage. A subsequent transfer-stream wait was redundant and is removed.

Every asynchronous tensor operation retains tensor owners (and INT8 scratch)
until its end event completes, even when profiling is disabled. Deferred event
collection queries rather than waits. This prevents memory-pool reuse after a
local intermediate Tensor is destroyed while CUDA still references it. An
unfinished operation scope uses a stream barrier on exceptions; teardown also
drains outstanding work. Scopes must not outlive their backend/event queue.

Required router readback, gather/scatter temporary-host-buffer waits, expert
lease release boundaries, transformer completion, and teardown waits remain.
No hot-path device-wide synchronization was found to remove. This is not a
globally asynchronous generation API, and tensor backends remain single-caller
execution objects (the event collector is independently mutex protected).

The implementation follows NVIDIA's CUDA 12.5 [stream dependency API](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-runtime-api/group__CUDART__STREAM.html)
and [event query/timing API](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-runtime-api/group__CUDART__EVENT.html).

## Reading profiling reports

Real-model reports add `cuda_kernel_time_ms` (sum of leaf operation event
spans), FP32/INT8 GEMM, expert GEMM, activation, attention-core, inclusive
attention, and inclusive expert execution times. Inclusive regions must **not**
be added to leaf totals: they overlap their child operations. CUDA event spans
can include launch gaps/interleaved stream work; these are not CUPTI
kernel-exclusive measurements. Legacy aggregate duration fields remain for
compatibility and may contain CPU wall timers and CUDA event spans. Use the
explicit `cuda_*` fields to isolate GPU work; CPU timings are never substituted
for missing CUDA measurements.

`cuda_memory_transfer_time_ms` sums existing tensor/expert backend H2D/D2H
copy event timings (not a complete device-to-device/KV-copy trace).
`synchronization_time_ms` sums host duration of backend stream/event
wait APIs, including their accounting overhead. Concurrent worker waits may
overlap each other or GPU execution: these categories are not an additive wall
clock decomposition. Startup/static copies are included in cumulative transfer
and synchronization metrics; token wall time excludes loading.

Implicit waits inside allocation/free, pageable copies, or cuBLAS host-scalar
reductions are outside this explicit wait counter. Nsight is required for a
complete synchronization timeline.

`cuda_leaf_span_wall_ratio` is an activity indicator, not SM utilization. It may
exceed one for concurrent operations/cumulative startup scopes. Hardware
`gpu_utilization_percent` stays null without a hardware sampler; use Nsight
Systems/Compute or external `nvidia-smi` sampling for real utilization and
occupancy. Reports never replace missing CUDA measurements with CPU numbers.

## Hermes qualification

Build native CUDA Release and Debug on Windows, preserving CUDA 12.5 `bin` on
PATH for runtime DLL discovery. Run all tests; Phase 23 must print no native
CUDA skip on the RTX 4070. Run Compute Sanitizer memcheck/racecheck on
`hypermoe_phase23_tests.exe` to validate shared-memory barriers and lifetime
handling; Mac ASan/UBSan does not instrument device kernels.

```powershell
ctest --test-dir build --output-on-failure -C Release
build/Release/hypermoe_int8_gemm_benchmark.exe phase23-gemm.json
build/Release/hypermoe_profile_real_model.exe <same-int8-artifact> <same-token-ids> cuda phase23-reference.json --expert-device-budget 2GiB --int8-gemm reference
build/Release/hypermoe_profile_real_model.exe <same-int8-artifact> <same-token-ids> cuda phase23-cooperative.json --expert-device-budget 2GiB --int8-gemm cooperative
```

The microbenchmark compares scalar/cooperative modes with profiling both off
and on, three warmups and twenty repetitions, identical explicit latency
boundaries, CPU correctness gates, scratch/launch counts, transfer and wait
time. It emits an empty measurement array when native CUDA is unavailable.
For the full model, use identical artifact/tokens/budgets/overlap settings and
fresh processes. Compare TTFT, tokens/sec, transfers, cache hits, VRAM, expert
GEMM, attention, and wait times. Also compare CPU/CUDA traced outputs using the
existing real-model validation tool. Historical 0.158 tokens/sec is not a
substitute for a same-workload reference run.

Current limits: no local native CUDA build or Qwen3 checkpoint on Mac; native
kernel correctness/performance needs Hermes qualification. Serial attention,
router readback, gather/scatter waits, FP32 static GEMM, launch/event overhead,
and residency misses remain possible bottlenecks. Next phase should follow the
measured timeline, rather than introduce graphs or precision changes blindly.
