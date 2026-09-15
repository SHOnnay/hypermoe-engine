# Adaptive expert residency (Phase 22C)

## Existing lifecycle and authority

CPU demand stops at NVMe → RAM → CPU. CUDA demand uses NVMe → RAM/pinned
staging → VRAM → CUDA. `TransferManager` owns the existing transfer/prefetch
streams, pinned staging and event-completed futures. `CudaTensorBackend` owns
compute. Phase 22B lookahead and availability/lease boundaries are unchanged.

`ExpertManager` owns resident buffers; `MemoryManager` enforces expert-only
limits. `prepareResidency` evicts through the selected cache policy before a
load; adoption performs the checked final allocation. Active leases exclude
executing experts. VRAM victims move to RAM when possible; RAM victims return
to NVMe by releasing buffers, not rewriting immutable weights. Scheduler
prediction remains a separately bounded speculative warm RAM cache.
`MemoryPressureController` retains its logical-margin behavior: no second
transfer system or hardware pressure controller is introduced.

## Manual and automatic budgets

Manual defaults remain **512MiB expert VRAM / 2GiB expert RAM**. Existing
`expertDeviceBudgetBytes` and `expertRamBudgetBytes` accept positive byte counts.
The benchmark accepts 512MiB/1GiB/2GiB/4GiB; MB/GB are decimal. Manual mode
retains Phase 22B's preflight: reject device budgets exceeding free VRAM after
static loading or unable to hold the largest expert. Manual capacity is not
total VRAM; leave execution headroom.

Opt in with `automaticExpertDeviceBudget = true` (CUDA only). A pure planner
runs after static tensors and device/context initialization:

```text
available = min(observed free VRAM, total VRAM - static execution bytes)
reserved = KV + workspace + staging/free-pool + safety + transfer allowance
expert budget = round_down_256(available - reserved)
```

Free VRAM already excludes static allocation: static bytes are not subtracted
twice. Other GPU users lower available capacity. Auto mode uses safe remaining
memory, not the configured manual value as a cap; both are reported separately.

Default auto reservations: 256MiB KV, 512MiB workspace, 512MiB staging/free pool,
512MiB safety. The staging minimum references the existing pool's free-block
retention limit. A separate allowance covers an aligned largest expert per
transfer worker and worst-case per-expert alignment padding. Checked arithmetic
rejects exhausted headroom, overflow and experts that cannot fit before expert
allocation. Reservations are envelopes, not eager allocations.

Auto-mode caches must be created by this runtime. Their summed maximum sizes,
each doubled for old/new buffers coexisting during growth, must fit the KV
envelope. Weak ownership releases capacity on cache destruction; external caches
are rejected in auto mode. The forward path checks a conservative workspace
envelope derived from wide QKV/GQA dimensions, attention scores, retained
layer/expert outputs, logits and INT8 conversion scratch. Increase reservations
for larger sequences or retained diagnostic traces.

This is safe startup sizing, not continuously elastic allocation. No process
can guarantee against another process allocating VRAM after the sample, driver
changes, or arbitrary caller-retained forward results/tensors. Avoid competing
GPU workloads and release traces when finished. Existing CUDA allocation errors
are surfaced, not hidden or bypassed.

## Existing predictor, better residency decisions

Packed adaptive residency selects age-aware `HybridPolicy`. The existing
`MoERuntime` callback supplies expected layer/expert probability and confidence;
no predictor or model format changes. Its clock follows expert accesses and
residency registrations, not wall time:

```text
decay(age) = 2 ^ (-age / 4096)
frequency = lazily decayed access count
score = 0.4 * frequency/(frequency+8)
      + 0.3 * access recency decay
      + 0.3 * probability * confidence * prediction-age decay
```

Hot experts resist one-off recency. Newly hot experts can compete with stale
lifetime favorites. Low-confidence/stale hints cannot permanently protect cold
experts. Protection is a score, not an unbounded pin; equal scores use
deterministic policy-ID ordering. `adaptiveResidency = false` retains LRU.
Standalone `HybridPolicy()` retains its Phase 21 formula; age-aware mode is
explicit. Residency diagnostics expose the actual policy score.

## Profiling and tests

JSON preserves existing fields and adds configured/effective budget distinction,
auto mode, reservation/allowance bytes, promotions, physical expert device bytes
(including retained pool blocks), and sampled total/free GPU memory.
`expert_device_budget_bytes` is the enforced effective capacity;
`configured_expert_device_budget_bytes` is the manual request.
`resident_expert_device_bytes` and resident counts are manager-owned weights.
`vram_promotions` counts successful non-VRAM → VRAM transitions, not hits;
`vram_evictions` counts departures. `nvme_transfer_bytes` counts requested expert
regions including prefetch, not physical SSD I/O (mmap may hit OS cache).
`ram_to_vram_transfer_bytes` aliases the expert transfer backend's H2D bytes,
including cold-load RAM staging but excluding static tensor copies.
`vram_usage_bytes` remains the legacy static+resident+KV estimate, not telemetry;
sampled GPU usage includes other allocations/processes. GPU utilization remains
null: sample externally.

Always-on tests cover budgets, auto arithmetic, shared-memory headroom,
overflow/exhaustion, eviction/promotion, INT8 bytes/output, lease safety, aging,
and legacy defaults. Conditional CUDA tests repeat residency transfers and full
packed Qwen fixture correctness, cache overcommit and reservation release. These
are not real Qwen3-30B hardware tests.

## Hermes RTX 4070 qualification

Keep artifact, tokens, overlap and prediction settings identical. Record OS
cache state and repeat trials. Windows CUDA 12.5 `bin` must be on PATH for DLLs.

```powershell
foreach ($budget in @("512MiB", "1GiB", "2GiB", "4GiB")) {
    .\build-release\Release\hypermoe_profile_real_model.exe ARTIFACT 1,42,73 cuda "budget_$budget.json" --expert-device-budget $budget --expert-ram-budget 2GiB --overlap on
    if ($LASTEXITCODE -ne 0) { throw "Budget validation failed: $budget" }
}
.\build-release\Release\hypermoe_profile_real_model.exe ARTIFACT 1,42,73 cuda auto.json --auto-expert-device-budget on --kv-reservation 256MiB --workspace-reservation 512MiB --staging-reservation 512MiB --safety-reservation 512MiB --overlap on
```

Run Release/Debug full CTest and CPU/CUDA logits/intermediate oracle, then report
TTFT, throughput, NVMe/H2D bytes, cache hits, residents, promotions/evictions,
logical/physical VRAM, RSS and explicit synchronization counts. Sample GPU
utilization/timeline externally.

User-supplied Hermes baselines: Phase 22A INT8 0.118 tok/s, 42.4s latency,
8.70GB transfers, 4.01% hits; Phase 22B 0.089 tok/s, 55.6s latency, same transfers
and hit rate. Phase 22C real-model numbers require PC measurement.

`hypermoe_residency_benchmark cpu REPORT.json` measures actual loading/execution
of four 2x2 INT8 experts at one/two/four-expert RAM capacity with LRU/age-aware
Hybrid. CUDA mode requires native CUDA. Neither mode reports generation speed
or pretends these experts are Qwen3-30B weights.

One Mac AppleClang Release run on 2026-09-15, 100 expert-layer calls (not tokens):

| RAM capacity | Policy | Total latency | Cache hits | Requested loader bytes | RAM evictions |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 tiny expert | LRU | 7.791ms | 0.0% | 2,400 | 199 |
| 1 tiny expert | Age-aware Hybrid | 10.419ms | 0.0% | 2,400 | 199 |
| 2 tiny experts | LRU | 6.863ms | 25.5% | 1,788 | 147 |
| 2 tiny experts | Age-aware Hybrid | 6.506ms | 33.0% | 1,608 | 132 |
| 4 tiny experts | LRU | 3.165ms | 98.0% | 48 | 0 |
| 4 tiny experts | Age-aware Hybrid | 2.344ms | 98.0% | 48 | 0 |

Timings vary with OS scheduling; these establish no RTX inference speedup.
Capacity and transfer-count differences describe only this deterministic trace.
At one-expert capacity neither policy can preserve the working set; Hybrid was
slower in this run. Increasing capacity is the primary experiment, not a claim
that more elaborate scoring always makes execution faster.

## Mac implementation validation (2026-09-15)

AppleClang 21, C++20, strict warnings-as-errors: full Release and Debug builds
succeeded. Full CTest passed 25/25 in Release (30.10s), Debug (32.78s) and
ASan/UBSan Debug (60.16s), including Phase 20, 21, 22A, 22B and 22C. Sanitizer
execution used `ASAN_OPTIONS=detect_leaks=0` (Apple LeakSanitizer unavailable)
and `UBSAN_OPTIONS=halt_on_error=1`; this does not claim leak-sanitizer coverage.
Release's normal CUDA detection correctly selected CPU fallback when the toolkit
was absent; native CUDA cases were skipped explicitly. No real Qwen3 checkpoint
or RTX 4070 is available here. Real-model budgets, CUDA correctness and throughput
remain pending Hermes qualification, not certified by the Mac fixture.
