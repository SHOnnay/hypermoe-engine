# GPU expert execution

CUDA remains behind `TensorBackend`. `CudaTensorBackend` allocates pool-backed
device tensors, performs checked CPU-to-device, device-to-CPU, and device-to-device
copies, synchronizes persistent streams, and implements row-major FP32 matrix
multiplication with cuBLAS SGEMM. `ExpertMlpExecutor` uses the same interface for
CPU and CUDA, so gate, up, and down projections do not contain CUDA-specific
branches.

The Phase 11 correctness path is:

```text
ExpertStore → Scheduler → pinned staging → CudaMemoryPool
      → ExpertResidencyLease → TensorView projection slices
      → cuBLAS gate/up → SiLU → gated multiply → cuBLAS down
```

The residency lease spans all operations and the final backend synchronization.
`ExpertManager` excludes leased experts from eviction and rejects explicit moves
while a lease is active. Once synchronization completes and the lease is released,
the same device allocation may be demoted or returned to the pool.

FP32 is the correctness baseline. FP16 and BF16 storage select an explicit FP32
execution plan. Without native precision kernels, conversion stages only the
selected tensor through host reference conversion and uploads the FP32 result;
the complete model is never expanded. CUDA activation and elementwise operations
currently use the existing host-staged reference bridge. These are correct but
intentionally not described as optimized GPU kernels.
