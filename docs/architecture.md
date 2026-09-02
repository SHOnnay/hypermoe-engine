# HyperMoE architecture through Phase 3.5

The runtime separates durable storage, movement, residency, eviction policy,
hardware access, and measurement so future model adapters do not own memory
policy.

```text
router/model simulation
          │ demand or prediction
          ▼
 Scheduler + residency state machine ◄── RuntimeEvent subscribers
          │ coalesced priority request
          ▼
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
- `Scheduler` owns the expert-level priority queue and lifecycle. It coalesces
  duplicate requests, upgrades queued prefetch work to inference priority,
  submits transfers, fulfills shared futures, and publishes completion/failure
  events. Multiple consumers therefore share one physical movement.
- `ExpertResidencyStateMachine` validates transitions across `REQUESTED`,
  `QUEUED`, `LOADING`, `READY`, `IN_USE`, `EVICTING`, and `FAILED`, while tracking
  current/target tiers, last use, and usage count. Residency and pending work use
  the composite `(layer_id, expert_id)` identity required by per-layer MoE IDs.
- `Prefetcher` is a prediction interface. The baseline `LocalityPrefetcher`
  combines recent experts and next-layer workload hints into confidence-ranked
  requests without depending on any model architecture.
- `RuntimeEventBus` decouples scheduler transitions from cache, profiling, and
  future runtime consumers. Subscriber failures cannot corrupt scheduler state.
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
  timing, GPU-memory use, queue depth, byte counts, stalls, prefetch outcomes,
  average scheduler queue wait, and transfer overlap.
- `HardwareInfo` reports CPU, logical cores, RAM, available storage, CUDA
  build/runtime state, GPU name, VRAM, runtime version, and driver version.

## Ownership and synchronization

Mapped spans never outlive their `ExpertStore`. Loads return owning vectors.
Transfer tasks, shared futures, and cancellation flags use shared ownership until
their promises are fulfilled. Device and pinned buffers retain the backend that
must free them. Scheduler queue/pending state is protected independently from the
residency map; event callbacks run after event-bus locks are released. A transfer
task's priority is immutable after dispatch, preventing worker/request races.
`ExpertManager` serializes its existing ownership transitions, while
`MemoryManager` independently protects accounting.

The transfer future is fulfilled only after the backend completion event. A CUDA
device allocation therefore cannot become visible to a consumer while its copy is
incomplete.

## Deferred intentionally

- Model-format parsing and Qwen/GLM/DeepSeek/Kimi adapters
- Tensor routing, execution, token generation, and KV cache
- Direct-storage integrations and unbuffered platform-specific NVMe benchmarks
- Router-produced probabilities and measured compute/transfer overlap
- Automatic scheduler-driven capacity selection and eviction policy execution
- Custom CUDA kernels
