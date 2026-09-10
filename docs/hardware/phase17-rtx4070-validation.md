# HyperMoE Phase 17 CUDA Accelerated Inference — RTX 4070 Validation

**Date:** 2026-09-10  
**Scope:** Hardware validation only — no architecture changes, no feature implementation

---

## 1. Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4070 (Ada Lovelace, CC 8.9) |
| VRAM | 12,282 MiB total / 11,385 MiB free (idle) |
| CPU | Intel Core i5-13400F |
| RAM | 16 GB |
| OS | Windows (build 21996) |
| NVIDIA Driver | 555.85 |
| CUDA Toolkit | 12.5 (V12.5.40) |
| cuBLAS | 12.5 |
| MSVC | 19.44.35228 (VS 2022 BuildTools 17.14) |
| CMake | 4.4.3 |
| Windows SDK | 10.0.26100 |

---

## 2. Source Synchronization

- **Commit:** `5e3cdc3` "Implement CUDA accelerated inference runtime"
- **Branch:** `main` (clean, up to date with `origin/main`)
- **Phase 17 files verified:**
  - `src/runtime/cache/CudaKVCache.hpp`
  - `src/runtime/generation/InferenceConfig.hpp`
  - `src/transformer/attention/CudaAttention.cpp`
  - `src/transformer/attention/CudaAttention.hpp`
  - `tests/phase17_tests.cpp`

---

## 3. Build

| Config | Result | Details |
|--------|--------|---------|
| **Release (CUDA ON)** | ✅ **PASS** | 37 targets, 0 errors, 0 warnings |
| **Debug (CUDA ON)** | ✅ **PASS** | 37 targets, 0 errors, 0 warnings |

**CMake CUDA detection:** `HyperMoE CUDA: toolkit 12.5.40; runtime API and cuBLAS enabled`  
**Linking verified:** `cudart64_12.dll` + `cublas64_12.dll` bound into CUDA targets via `dumpbin /dependents`

---

## 4. Tests — 18/19 PASS (95%) in Release AND Debug

| Test | Result |
|------|--------|
| hypermoe_phase1–14 | ✅ Passed |
| hypermoe_platform | ✅ Passed |
| hypermoe_phase15_16 | ✅ Passed |
| **hypermoe_phase17** | ❌ **Failed (exit 1, both configs)** |

Reproduced directly: `hypermoe_phase17_tests.exe` exits 1 in Release and Debug with:

### Phase 17 Failure Analysis
```
FAIL: unexpected exception: forward pass state is inconsistent
```

**Root cause (test bug, not code bug):**  
The test's mock `CudaSessionModel::forward()` returns `ForwardPass` with an **empty `routing` vector `{}`.**  
`ForwardState::update()` validates `routing.size() == hiddenStates.rows()` (one routing decision per token). The mock violates this contract.

**Location:** `tests/phase17_tests.cpp:201` — `return {std::move(hidden), std::move(logits), 0, {}};`

**Fix required in test only:** populate `routing` with `tokenIds.size()` valid `RouterDecision` entries (matching hidden states batch dimension). No source changes needed.

---

## 5. GPU Benchmarks

### 5.1 GPU Inference Benchmark (`hypermoe_gpu_inference_benchmark.exe`)

| Metric | Value |
|--------|-------|
| GPU Initialization | **106.6 ms** (includes CUDA context + cuBLAS handle) |
| H2D Transfer | **0.368 ms** |
| **CUDA Expert Execution** | **6.49 ms** |
| CUDA Attention | **0.63 ms** |
| **Transformer Layer (end-to-end)** | **7.12 ms** |
| Throughput | **562 tokens/s** |
| VRAM Allocated | 0 MiB (benchmark uses statically sized test tensors) |

### 5.2 CUDA Runtime Benchmark (`hypermoe_cuda_runtime_benchmark.exe`)

| Metric | Value |
|--------|-------|
| Allocation Latency | 256 µs |
| **H2D Bandwidth** | **13.8 GiB/s** |
| **D2H Bandwidth** | **14.2 GiB/s** |
| GEMM (cuBLAS SGEMM, small) | 21.4 GFLOP/s |
| CPU Expert | 0.22 ms |
| CUDA Expert | 0.67 ms |
| Transfer/Compute Overlap | 49.6% |

### 5.3 Generation Benchmark (`hypermoe_generation_benchmark.exe`) — CPU incremental fixture

| Metric | Value |
|--------|-------|
| Avg Decode Step | **0.047 ms** |
| Throughput | **319,881 tokens/s** |
| Prefill | 0.0033 ms |
| KV Cache Peak | 1.84 KB |

### 5.4 CPU Model Runtime Benchmark (`hypermoe_model_runtime_benchmark.exe`)

| Metric (CPU) | Value |
|--------------|-------|
| 2-layer MoE Model | **0.091 ms** |
| Attention/Layer | 0.0085 ms |
| Expert/Layer | 0.0168 ms |

---

## 6. CPU vs CUDA Comparison

| Workload | CPU | CUDA | Notes |
|----------|-----|------|-------|
| **Expert execution (small)** | **0.22 ms** | 6.49 ms (gpu-inference fixture; 0.67 ms in cuda-runtime fixture) | CUDA kernel launch overhead dominates at tiny shapes |
| **Attention** | ~0.008 ms | 0.63 ms | `CudaAttention` falls back to CPU reference for RoPE + scores; only projections are on GPU |
| **Transformer Layer** | 0.091 ms (2-layer model) | 7.12 ms | Full GPU path not yet wired for end-to-end MoE |
| **Generation throughput** | **320K tok/s** | 562 tok/s | CPU fixture is optimized incremental; GPU benchmark is full transformer layer |

### Key Observations
1. **Expert execution is slower on CUDA** at current test shapes — expected for single small GEMM + launch overhead. Benefits appear at larger batch/token counts.
2. **Attention is not fully CUDA-accelerated** — `CudaAttention` projects via cuBLAS but copies scores/probs/context from CPU reference (RoPE + softmax remain on host). This adds 2× D2H + H2D per layer.
3. **Transformer layer timing includes H2D/D2H** for weights and intermediates — not just compute.
4. **Initialization cost (~107 ms)** is one-time (CUDA context + cuBLAS handle creation).

---

## 7. Bottleneck Profile

| Bottleneck | Evidence | Impact |
|------------|----------|--------|
| **Kernel launch overhead** | `cuda_expert_ms` (6.5 ms) vs `cpu_expert_ms` (0.2 ms) at tiny shapes | Dominates at batch=1, small hidden dim |
| **Partial GPU attention** | `CudaAttention` copies 4 tensors (scores, probs, context, RoPE) between host/device | 2–3× D2H+H2D per layer |
| **cuBLAS GEMM on small matrices** | 21.4 GFLOP/s vs 29 TFLOP/s peak | Launch-bound, not compute-bound |
| **VRAM pressure** | ~500–670 MiB used under load; benchmarks allocate ~50 MB | Not limiting yet; headroom for larger models |
| **CPU synchronization** | `backend->synchronize()` after each cuBLAS call | Serializes GPU work; no async pipelining |

### nvidia-smi During Benchmark (sampled during 6× gpu-inference loop)
```
GPU Utilization: 6–13%
Memory Utilization: 0–7%
Memory Used: 498–669 MiB (511 MiB idle)
Temperature: 43–44°C
```
GPU is **heavily underutilized** — work is memory/launch bound, not compute bound.

---

## 8. Recommendations (Measurement-Only — No Optimization)

1. **Phase 17 test fix:** Update `tests/phase17_tests.cpp` mock `CudaSessionModel::forward()` to return valid routing decisions matching `tokenIds.size()`. This unblocks CI without touching runtime code.

2. **Attention path:** Full CUDA attention (RoPE + softmax + weighted sum on GPU) would eliminate 4 tensor copies per layer.

3. **Batch inference:** Benchmark with larger batch sizes (e.g., 32–128 tokens) where cuBLAS saturates SMs and launch overhead amortizes.

4. **Async pipelining:** Overlap H2D/D2H with compute using separate streams (`Transfer` vs `Compute`) — `CudaTensorBackend` already has `CudaStreamManager` with 3 roles.

5. **Fused expert kernel:** Current `ExpertMlpExecutor` issues separate cuBLAS calls per projection. A custom kernel fusing gate/up/down would reduce launches.

6. **Persistent CUDA context:** For server workloads, keep `CudaTensorBackend` alive across requests to amortize ~107 ms initialization.

---

## 9. Readiness for Phase 17 Validation

| Gate | Status |
|------|--------|
| CUDA Toolkit + cuBLAS | ✅ Ready |
| Build (Release/Debug, CUDA ON) | ✅ Ready |
| Phase 1–16 regression | ✅ 18/18 pass (both configs) |
| Phase 17 CUDA components exist | ✅ Verified on disk |
| GPU inference benchmarks | ✅ Measured (baseline captured) |
| CPU vs CUDA comparison | ✅ Measured (baseline captured) |
| nvidia-smi profiling | ✅ Captured |
| **Test blocker (phase17_tests.cpp)** | ⚠️ **One test fix needed** |

---

## 10. Files Generated

- `docs/hardware/phase17-rtx4070-validation.md` (this report)
- `build/Release/gpu_inference_report.json` — GPU inference benchmark
- `build/Release/cuda_runtime_report.json` — CUDA runtime benchmark
- `build/Release/generation_report.json.report` — Generation benchmark
- `build/Release/model_runtime_report.json` — CPU model runtime benchmark
- `build/Release/end_to_end_report.json` — End-to-end benchmark
- `build/Release/hardware_report.json` — System hardware benchmark