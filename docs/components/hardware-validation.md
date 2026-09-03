# NVIDIA hardware validation

`hypermoe_cuda_runtime_benchmark` is the Phase 11 hardware benchmark. It records:

- direct VRAM allocation latency;
- pinned host-to-device and device-to-host bandwidth;
- FP32 cuBLAS GEMM throughput;
- CPU and CUDA expert MLP latency;
- concurrent demand/prefetch transfer wall time and observed overlap;
- scheduler queue wait and average transfer latency;
- current/peak backend allocation, resident experts, and eviction events.

Run it on the target machine with:

```sh
./build/hypermoe_cuda_runtime_benchmark cuda_runtime_report.json
```

The benchmark first runs `CudaRuntimeValidator`. If CUDA was not compiled or no
NVIDIA device is usable, it emits a valid report with `"skipped": true` and an
explicit reason; GPU numbers are not estimated. Results depend on driver state,
power limits, PCIe topology, thermals, buffer size, and other system load and must
be captured on the RTX 4070 host before choosing runtime defaults.

`cuda_real_expert_tests` uses a format-correct Qwen3 MoE SafeTensors artifact. It
imports and packs the artifact, routes a token, uploads the selected contiguous
expert through pinned memory, executes the projection chain, and compares router
logits, top-k selection, every intermediate expert stage, and final output against
the independent CPU oracle at FP32 tolerance. It also proves that an active
residency lease blocks eviction. A full public checkpoint remains a separate
target-hardware validation step.
