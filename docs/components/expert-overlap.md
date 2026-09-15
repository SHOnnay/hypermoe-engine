# Expert transfer overlap and residency budgets

Phase 22B contains only transfer/compute overlap and configurable expert budgets.
INT8 packing, projection scales, model loading, attention layout, GQA and CPU
fallback remain the existing contracts. Architecture validation accepts wide
attention and independent projection-head dimensions; it does not infer them
from hidden width.

## Ownership and synchronization

- TransferManager workers own NVMe range loads, pinned staging, the H2D/D2H
  transfer stream and speculative prefetch stream. They record a CUDA event
  after each copy and wait for that event before publishing a completed result.
  This worker wait does not wait on the tensor backend's compute stream.
- CudaTensorBackend owns compute streams and cuBLAS execution. MoERuntime keeps
  the current expert lease through the existing synchronous scatter completion
  (or output readback on fallback) without adding a second compute wait, and submits
  one next expert first, when the residency budget can hold both payloads.
- ExpertManager makes room through the existing eviction policy before the
  lookahead load. Its preparation API is for the serialized runtime scheduling
  consumer; it is not a new concurrent reservation allocator. Adoption performs
  the final checked reservation. Active leases cannot be evicted.
- Single-expert budgets use serial loading. CPU stops in RAM and can overlap
  NVMe loading with CPU computation. Native CUDA transfers stop at event-ready
  VRAM. No expert is exposed during an incomplete transfer.
- If B has an unfinished warm prediction, A does not wait for it before
  computing. That warm load continues independently and is consumed at B's
  execution boundary.
- Existing GEMM profiling, routing readback, gather/scatter and materialization
  waits are retained. CUDA allocation/free may introduce driver-level waits;
  this phase does not claim zero synchronization or fully asynchronous compute.

Scheduler references are removed after adoption, and consumed futures are
discarded. Otherwise a demoted expert could remain physically allocated outside
manager accounting. Packed-runtime prediction stages into a bounded speculative
RAM cache; demand consumption adopts it without a second disk read. Cache entries
are expired/trimmed deterministically and scheduler location metadata reconciles
with manager-owned residency. Standalone scheduler prefetch defaults are unchanged.

## Budget configuration

`PackedRuntimeConfiguration` already had these fields; their defaults remain:

```cpp
expertDeviceBudgetBytes = 512 * 1024 * 1024;
expertRamBudgetBytes = 2ULL * 1024 * 1024 * 1024;
transferComputeOverlap = true;
```

Budgets smaller than the largest executable expert fail explicitly. On CUDA a
budget above currently free VRAM after static tensor loading fails explicitly.
This check is not a reservation of future KV/activation storage: leave headroom.
Residency budgets exclude static tensors, KV cache, activation tensors, pinned
staging and pool alignment/free blocks. The existing transfer pool can retain up
to 512 MiB free blocks. Speculative RAM cache has its own upper bound equal to
the configured RAM budget, in addition to manager residency and in-flight worker
staging. Do not treat logical residency bytes as total process RSS/VRAM.

```sh
hypermoe_profile_real_model ARTIFACT 1,42,73 cuda serial_512m.json \
  --expert-device-budget 512MiB --expert-ram-budget 2GiB --overlap off
hypermoe_profile_real_model ARTIFACT 1,42,73 cuda overlap_2g.json \
  --expert-device-budget 2GiB --expert-ram-budget 2GiB --overlap on
```

Use 512MiB, 1GiB, 2GiB and 4GiB for a binary-budget sweep; `MB`/`GB` use decimal
units. Bytes without a suffix are accepted. Invalid, zero or overflowing values
fail rather than wrapping. Keep token IDs, artifact and all other settings fixed.

Reports include TTFT/throughput, budget bytes, resident expert counts, manager
evictions, actual NVMe read bytes, expert H2D bytes, cache hit rate and explicit
backend synchronization/event-wait counts. GPU utilization is `null` because
no hardware utilization sampler is installed; collect it externally on the PC.
Timing-event waits used for cuBLAS profiling are included in synchronization
counts. Driver allocation waits and implicit cuBLAS host-scalar waits are not.
NVMe volume includes speculative loads; H2D volume is reported separately. These
NVMe bytes count requested expert regions from the loader, not physical SSD I/O;
mmap reads can be serviced by the OS page cache. The
actual counters replace the older cache-miss-derived volume approximation, so
compare identical new-runtime serial/overlap runs as well as the Phase 22A report.

## Validation and benchmark scope

`hypermoe_phase22b_tests` checks ordering with a condition-variable handshake,
INT8/FP32 serial-vs-overlap CPU results, warm ownership, configured eviction and
promotion, protected leases, wide architecture metadata and optional native
CUDA INT8 output correctness under eviction. CPU-backed device buffers provide
always-on manager budget tests; they are not GPU performance tests.

`hypermoe_expert_pipeline_benchmark cpu REPORT` measures the production pipeline
on a four-expert, 2x2 INT8 fixture at one/two/four-expert capacity. Its calls/sec
are not generated tokens/sec. `cuda` mode requires native CUDA and never silently
substitutes CPU numbers. Real Qwen3-30B-A3B and RTX 4070 are unavailable on the Mac.

One Apple Silicon Release run on 2026-09-15 measured 100 expert-layer calls:

| RAM capacity | Serial total | Lookahead total | Loader bytes, either mode | Cache hits, either mode |
| --- | ---: | ---: | ---: | ---: |
| 2 tiny experts | 3.123 ms | 3.038 ms | 1,212 | 49.5% |
| 4 tiny experts | 1.841 ms | 1.827 ms | 48 | 98.0% |

These small timings vary with OS scheduling; other runs had slower lookahead.
They establish no consistent speedup and no RTX inference performance claim.
Capacity changes and transfer accounting are deterministic for this trace.
The supplied Hermes Phase 22A baseline is TTFT 42.4 s, 0.118 tokens/sec,
8.70 GB expert transfers and 4.01% cache hits. Phase 22B real-model metrics are
reported by Hermes as 0.089 tokens/sec, 55.6s latency with the same 8.70GB
transfers and 4.01% cache hits. The microbenchmark is not a substitute. Phase
22C adds [opt-in adaptive capacity](adaptive-residency.md), not another transfer
implementation.

Hermes qualification: build native CUDA Release/Debug, run all tests, compare
serial vs overlap logits on the same INT8 artifact, then sweep budgets and record
TTFT, tokens/sec, disk/H2D volume, cache hits, evictions, physical VRAM/RSS,
resident experts, synchronization count, and externally sampled GPU utilization.
Use a CUDA timeline to establish actual H2D/compute overlap; no speedup claim is
made before that measurement. Windows CUDA 12.5 runtime DLLs require its `bin`
directory on PATH. No source workaround is introduced.
