# Real expert execution

Phase 9's first artifact-backed path is:

```text
SafeTensors → QwenImporter → ModelManifest → ExpertPacker
            → ExpertStore → Scheduler → ExpertManager
            → ExpertResidencyLease → TensorView slices → ExpertMlpExecutor
```

`ExpertStore` maps `experts.bin` and exposes a checksummed expert range. The
scheduler loads that exact indexed range through `DiskLoader` and
`TransferManager`. `ExpertManager::adoptDeviceWeights` accounts for the returned
buffer without performing a second load. `ExpertWeightMap` converts manifest
projection offsets into checked, zero-copy gate/up/down views over the payload.

An `ExpertResidencyLease` is required around real execution. Acquiring it pins
the logical allocation and holds the physical `DeviceBuffer`. Cache candidates
exclude leased experts, and direct movement fails while a lease is active. The
runtime holds the lease until all backend work in `ExpertMlpExecutor` has
synchronized, then releases both manager and scheduler use state. Tensor views
remain non-owning; callers must not retain a view beyond its lease.

The first correctness target is one FP32 CPU expert using
`down(SiLU(input × gate) * (input × up))`. CUDA buffers and cuBLAS remain layout
compatible, but CUDA is not a Phase 9 correctness requirement. Full transformer
blocks, attention, tokenization, generation, grouped experts, and a server are
outside this milestone.
