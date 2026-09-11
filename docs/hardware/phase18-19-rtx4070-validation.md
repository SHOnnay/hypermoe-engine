# HyperMoE Phase 18/19 RTX 4070 Validation

**Date:** 2026-09-10 · **Scope:** hardware validation only — no architecture or runtime changes
**Commit validated:** `7cb469a` "Optimize CUDA dataflow and add real model validation" (+ local fixes `6201b4f`, `6ede77f`)

---

## 1. Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4070 (Ada, CC 8.9), 12,282 MiB |
| CPU / RAM | i5-13400F / 16 GB |
| OS | Windows (build 21996) |
| Driver | 555.85 |
| CUDA Toolkit | 12.5 (V12.5.40) — **native nvcc kernel compilation active** |
| MSVC | 19.44.35228 · CMake 4.4.3 · Ninja 1.12.1 (new build dir: `build-ninja/`) |

## 2. Toolchain Fix (required for native kernels)

The VS-generator configure reported `CMAKE_CUDA_COMPILER: NOTFOUND`, silently skipping `CudaKernels.cu` (tests printed `SKIP: native CUDA kernels unavailable`). Two fixes:

1. **Ninja generator** (`-G Ninja` with cl.exe host) — VS generator cannot locate nvcc without the CUDA VS integration; Ninja drives nvcc directly. Result: *"HyperMoE CUDA kernels: compiler 12.5.40 enabled"*.
2. **`CudaKernels.cu` include fix (`6201b4f`)** — `CUDART_INF_F`/`CUDART_NAN_F` require `<math_constants.h>` under native nvcc. One-line include; no logic change.

Post-fix test output has **no SKIP** — native kernels verified active.

## 3. Tests — 20/20 (100%)

All phases 1–17 + `hypermoe_phase18_19` pass in both VS/Release-Debug and Ninja/Release builds. `hypermoe_phase18_19_tests.exe` exit 0, no SKIP, with native kernels compiled in.

## 4. GPU Benchmarks (native-kernel build, `build-ninja/`)

### gpu_inference (schema v2 — new fields from Phase 18)
| Metric | Value |
|--------|-------|
| native_cuda_kernels | **true** |
| GPU init | 123.7 ms |
| H2D transfer | 0.328 ms |
| CUDA expert | 6.49 ms |
| **CUDA attention (native kernel)** | **0.113 ms** (was 0.63 ms pre-Phase-18) |
| Transformer layer | 6.60 ms |
| Throughput | 606 tok/s (was 545–562) |
| Measured CUDA transfer | 0.097 ms |
| Device-resident compute | 6.60 ms |
| Sync points | 23 |
| VRAM (benchmark) | 0.46 MiB |

### Other suites
| Benchmark | Key results |
|-----------|------------|
| cuda_runtime | H2D 13.4 GiB/s · D2H 14.4 GiB/s · GEMM 18.0 GFLOP/s · CUDA expert 0.86 ms · overlap 49.6% |
| generation (CPU fixture) | 150K tok/s decode · 0.10 ms/step |
| end_to_end | 768K routing tok/s · prefetch 74.96% · cache hit 88.17% |
| pipeline | transfer-hidden 81.7% · peak queue 3 · async stalls 15,450 ms (fixture wall-time dominated) |
| model_validation | scan 1.10 ms · packing 3.15 ms · 2 shards/3 tensors/28 params · 4,128 B packed |
| real_expert | import 0.67 ms · pack 1.98 ms · execute 0.012 ms |
| transformer | CPU block 0.053 ms · 149K tok/s · CUDA layer path "interface-ready" in this suite |
| hardware | NVMe 1.84 GiB/s seq · pinned 13.8 GiB/s |

## 5. CPU vs CUDA

| Workload | CPU | CUDA (native kernels) |
|----------|-----|----------------------|
| Expert (small fixture) | 0.217 ms | 6.49 ms (gpu-inference fixture) / 0.86 ms (runtime fixture) |
| Attention | 0.0085 ms/layer | **0.113 ms** — 5.6× faster than pre-18 CUDA path; still launch-bound vs CPU at tiny shapes |
| Transformer layer | 0.091 ms (2-layer model) | 6.60 ms |

## 6. Bottlenecks (nvidia-smi during 3× benchmark loop)

- GPU utilization 3–13%, 604 MiB used, 45°C — **launch-bound**, not compute-bound
- 23 synchronization points per benchmark pass (Phase 18 reduced these from per-op to per-boundary; further reduction = async queue deepening)
- Small-shape GEMM (18 GFLOP/s vs ~29 TFLOP/s peak) — launch overhead dominates
- Expert path still pays per-call sync in fixtures

## 7. Recommendations (measurement only — no optimization performed)

1. Batch shapes up (32–128 tokens) to amortize launch overhead before judging kernel quality.
2. Reduce sync points below 23 (single final materialization per step).
3. Persistent backend across requests amortizes 124 ms init.
4. Real-checkpoint validation (TTFT/gen-speed fields are `null` pending a runnable Qwen fixture) — needs a checkpoint artifact supplied.
5. Native-kernel build should become the default qualification path (`build-ninja/` workflow documented here).

## 8. Status

| Gate | Result |
|------|--------|
| Native CUDA kernels compiled & active | ✅ |
| 20/20 tests (both build systems) | ✅ |
| GPU inference v2 metrics captured | ✅ |
| CPU-vs-CUDA comparison | ✅ |
| nvidia-smi profile | ✅ |
| Source changes | 1-line include fix + gitignore only (`6201b4f`, `6ede77f`, pushed) |
