# HyperMoE Hardware Validation — RTX 4070

**Date:** 2026-09-10 · **Scope:** hardware preparation & validation only (no runtime/architecture changes)

## Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4070 (Ada Lovelace, CC 8.9) |
| VRAM | 12 GB (12,282 MiB total, 11,632 MiB free at validation) |
| CPU | Intel Core i5-13400F |
| RAM | 16 GB |
| OS | Windows (build 21996) |

## Software

| Component | Version |
|-----------|---------|
| MSVC | 19.44.35228 (VS 2022 BuildTools 17.14) |
| CMake | 4.4.3 |
| Windows SDK | 10.0.26100 |
| NVIDIA Driver | 555.85 |
| CUDA Toolkit | 12.5 (V12.5.40) — nvcc, cuBLAS, cudart |
| Install path | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5` |

## CUDA Status

- **Toolkit:** installed and verified (`nvcc --version` → Cuda compilation tools, release 12.5, V12.5.40)
- **Detection:** CMake reports `HyperMoE CUDA: toolkit 12.5.40; runtime API and cuBLAS enabled`
- **Linking:** `dumpbin /dependents` confirms `cudart64_12.dll` + `cublas64_12.dll` bound into CUDA targets
- **Runtime:** validation PASSED — device properties, compute capability, VRAM, versions (runtime=driver=12.50), 3 streams, event timing all green

## Build Status

| Config | Result |
|--------|--------|
| Release (CUDA ON) | ✅ 37 targets, warning-clean |
| Debug (CUDA ON) | ✅ 37 targets, warning-clean |

One required source fix (portability, no logic change): `src/tensor/backend/CudaTensorBackend.cpp` — `cudaMemcpyAsync` requires an explicit `static_cast<cudaStream_t>(stream)` from `hypermoe::backend::StreamHandle` (void*).

## Test Results (both configs, 18/18 = 100%)

Phases 1–14, phase15_16, platform, scheduler, phase2-integration — all passed.
CUDA-specific: **phase4** ✅, **phase5** ✅, **phase11** (real CUDA expert tests) ✅.

## GPU Benchmarks

| Metric | Result |
|--------|--------|
| H2D bandwidth (pinned) | **14.38 GiB/s** |
| H2D bandwidth (pageable) | 7.68 GiB/s |
| D2H bandwidth | **13.60 GiB/s** |
| GEMM baseline (cuBLAS SGEMM, small matrices) | 19.6 GFLOP/s |
| CUDA init / empty-stream event interval | 0.0165 ms |
| Allocation latency | 247 µs |
| CUDA expert forward | 1.06 ms |
| CPU expert forward (baseline) | 0.067 ms |
| Transfer↔compute overlap | 49.6% |
| VRAM allocated in benchmark | 48 MiB (peak 48 MiB) |
| CPU copy / RAM write | 10.96 / 12.62 GiB/s |
| NVMe seq / random (buffered) | 1.59 GiB/s / 839 MiB/s |

Full machine-readable results: `hardware_cuda_report.json` (repo root).

## Limitations

1. **GEMM baseline is small-matrix**: 19.6 GFLOP/s reflects tiny benchmark shapes dominated by launch overhead; RTX 4070 FP32 peak is ~29 TFLOPS on large matrices.
2. **CUDA expert slower than CPU** (1.06 ms vs 0.067 ms): expected — single small expert + per-call launch overhead, no kernel fusion yet. This is a workload-shape limitation, not a defect.
3. **DLL path dependency**: CUDA executables require `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin` on `PATH` (or DLLs beside the exe) or Windows fails the load with a missing-`cudart64_12.dll` error popup. Test runners include this path.
4. **No custom kernels**: HyperMoE uses runtime API + cuBLAS only; no nvcc kernel compilation is involved in the build.
5. **12 GB VRAM cap**: expert-residency policies must respect ~11.6 GiB usable (WDDM reserves part).

## Environment Notes

- Visual Studio CUDA integration: not required for this build (no .cu compilation); CMake's `find_package(CUDAToolkit)` + `CUDA::cudart`/`CUDA::cublas` targets handle everything.
- Compute capability 8.9 (Ada) confirmed via `nvidia-smi --query-gpu` and CUDA device properties.
