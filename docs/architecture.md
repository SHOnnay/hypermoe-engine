# HyperMoE architecture through Phase 3

The runtime separates durable storage, movement, residency, eviction policy,
hardware access, and measurement so future model adapters do not own memory
policy.

```text
experts.bin + experts.index
          │ checked range or mmap copy
          ▼
     DiskLoader
          │ prioritized, cancellable work queue
          ▼
       RAM buffer
          │ measured host copy
          ▼
  PinnedBuffer (cudaHostAlloc or aligned fallback)
          │ ComputeBackend::copyToDevice
          ▼
 DeviceBuffer (cudaMalloc or CPU allocation)
          │ stream + completion event
          ▼
       future consumer
```

## Components

- `ExpertIndex` parses a versioned, fixed-width little-endian format and builds
  an O(1) composite layer/expert lookup table.
- `MappedFile` owns a read-only OS mapping and exposes checked subspans without
  reading the complete model into an application buffer.
- `ExpertStore` validates record bounds and CRC32 and supports mmap-backed or
  explicit range reads of one expert.
- `DiskLoader` returns an owning buffer synchronously or asynchronously.
- `TransferManager` bounds concurrency with worker threads, prioritizes queued
  loads, exposes futures, supports cooperative cancellation and callbacks, and
  completes device transfers through backend streams and events.
- `ComputeBackend` is the capability boundary. `CpuBackend` is always usable;
  `CudaBackend` is compiled only when CUDAToolkit is detected and also checks for
  a runtime device.
- `PinnedBuffer` uses `cudaHostAlloc` when CUDA is active and aligned ordinary
  memory otherwise. `DeviceBuffer` ties allocation lifetime to its backend.
- `MemoryManager` enforces logical VRAM/RAM limits with allocation IDs.
- `ExpertManager` remains the sole logical residency owner. When attached to a
  transfer manager, cold movement owns checked weight bytes.
- `CachePolicy` has interchangeable LRU, LFU, and hybrid implementations. The
  hybrid normalizes signals before applying 0.4 frequency, 0.3 recency,
  0.2 layer probability, and 0.1 prefetch confidence.
- `MemoryPressureController` restores configurable safety margins and monitors
  transfer-queue depth.
- `Profiler` collects requests, transfers, evictions, pressure, CUDA/NVMe/RAM
  timing, GPU-memory use, queue depth, byte counts, and stalls.
- `HardwareInfo` reports CPU, logical cores, RAM, available storage, CUDA
  build/runtime state, GPU name, VRAM, runtime version, and driver version.

## Ownership and synchronization

Mapped spans never outlive their `ExpertStore`. Loads return owning vectors.
Transfer tasks and cancellation flags use shared ownership until their promises
are fulfilled. Device and pinned buffers retain the backend that must free them.
`ExpertManager` serializes residency transitions, while `MemoryManager`
independently protects accounting. Lock acquisition flows from expert ownership to
memory accounting; the reverse direction is not used.

The transfer future is fulfilled only after the backend completion event. A CUDA
device allocation therefore cannot become visible to a consumer while its copy is
incomplete.

## Deferred intentionally

- Model-format parsing and Qwen/GLM/DeepSeek/Kimi adapters
- Tensor routing, execution, token generation, and KV cache
- Direct-storage integrations and unbuffered platform-specific NVMe benchmarks
- Overlapped prefetch/inference orchestration
- Custom CUDA kernels
