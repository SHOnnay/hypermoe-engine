# HyperMoE architecture through Phase 7

The runtime separates durable storage, movement, residency, eviction policy,
hardware access, and measurement so future model adapters do not own memory
policy.

```text
model manifest → ModelAdapter → capabilities + neutral metadata
          │
          ▼
 hidden state + router weights → Router → RouterDecision
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
          │ persistent transfer/prefetch stream
          ▼
 CudaMemoryPool-backed DeviceBuffer
          │ stream + completion event
          ▼
 Tensor owner / non-owning TensorView slices
          │ TensorBackend
          ▼
 ExpertWeightMap → gate/up GEMMs → activation → down GEMM
          │ routing-score weighted combination
          ▼
    generic MoE layer output
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
- `CudaRuntime` initializes a selected device, reports compute capability and
  live VRAM information, owns created streams/events, and shuts them down after
  synchronization. With CUDA disabled it remains queryable and reports
  unavailable without including CUDA headers in public interfaces.
- `CudaStreamManager` owns persistent compute, transfer, and prefetch streams.
  Transfer workers select transfer or prefetch by request priority instead of
  creating a stream for every expert.
- `CudaMemoryPool` caches 256-byte-aligned device blocks and uses best-fit reuse.
  Its control state can outlive the pool facade, so outstanding RAII buffers do
  not lose the backend needed for safe release.
- `PinnedBuffer` uses `cudaHostAlloc` when CUDA is active and aligned ordinary
  memory otherwise. `DeviceBuffer` ties allocation lifetime to its backend.
- `MemoryManager` enforces logical VRAM/RAM limits with allocation IDs.
- `ExpertManager` remains the sole logical residency owner. When attached to a
  transfer manager, cold movement owns checked weight bytes. A resident backend
  buffer can be exposed as a tensor only when shape/dtype metadata matches its
  exact stored byte size.
- `ExpertManager` keys residency by `(layer_id, expert_id)`, allowing model-local
  expert IDs to repeat across layers. Existing single-ID APIs remain valid only
  when the registered ID is unambiguous.
- `Shape` stores dimensions and element strides with checked element-count and
  addressable-span arithmetic. `Tensor` adds dtype, device ordinal, logical byte
  count, backing byte count, pointer, and shared RAII ownership.
- `TensorView` copies only metadata and holds a weak lifetime token. It provides
  checked read-only/writable access and byte-offset slices over tensor,
  `DeviceBuffer`, and resident expert storage without extending residency between
  operations. Backends promote the weak token for the duration of each operation.
- `QuantizedTensor` validates contiguous INT8 and packed signed-Q4 storage,
  positive finite scale, signed zero point, device metadata, and versioned JSON
  serialization metadata. Packed Q4 rounds odd element counts up to one byte.
- `TensorBackend` defines allocation, copy, matmul, add, multiply, reshape, and
  synchronization. `CpuTensorBackend` is the reference implementation;
  `CudaTensorBackend` uses cuBLAS for row-major FP32 GEMM when CUDA is enabled.
- `MatmulExpertExecutor` retains the Phase 5 single-projection API.
  `ExpertMlpExecutor` composes gate/up/down FP32 projections, SiLU or exact GELU,
  and gated elementwise multiplication without embedding router or transformer
  behavior. CUDA projection uses cuBLAS; activation and multiply currently use
  explicit host staging.
- `ModelAdapter` converts format-specific metadata into neutral tensors, layers,
  router configuration, capabilities, and generic expert mappings. Runtime code
  never parses a tensor name or switches on `ModelArchitecture`.
- `QwenMoEAdapter` validates the HyperMoE v1 manifest and isolates Qwen-style
  projection/router name recognition. No checkpoint-native format is inferred.
- `ExpertWeightMap` maps layer/expert projection roles to tensor metadata and
  builds checked zero-copy views relative to the loaded expert payload.
- `CpuRouterBackend` implements FP32 scoring, stable softmax or raw scores,
  deterministic top-k for arbitrary k, and optional selected-score
  renormalization. `Router` is the backend/configuration boundary.
- `MoERuntime` coordinates route → parallel scheduling → buffer adoption → expert
  execution → routing-weighted combination for one MoE layer. Its current
  execution lock deliberately serializes layer calls while the contracts mature.
- `ExpertHistory` records per-layer expert frequency, cross-layer transitions,
  and the previous selection without implementing a predictor.
- `CachePolicy` has interchangeable LRU, LFU, and hybrid implementations. The
  hybrid normalizes signals before applying 0.4 frequency, 0.3 recency,
  0.2 layer probability, and 0.1 prefetch confidence.
- `MemoryPressureController` restores configurable safety margins and monitors
  transfer-queue depth.
- `Profiler` collects requests, transfers, evictions, pressure, CUDA/NVMe/RAM
  timing, GPU-memory use, queue depth, byte counts, stalls, prefetch outcomes,
  average scheduler queue wait, transfer overlap, kernel/matmul/expert/projection/
  activation/quantization time, tensor allocations, and externally supplied
  GPU-utilization observations.
- `HardwareInfo` reports CPU, logical cores, RAM, available storage, CUDA
  build/runtime state, GPU name, VRAM, runtime version, and driver version.

## Ownership and synchronization

Mapped spans never outlive their `ExpertStore`. Loads return owning vectors.
Transfer tasks, shared futures, tensors, and cancellation flags use shared
ownership until their promises are fulfilled. Tensor views deliberately do not;
the scheduler must retain or pin residency through execution. Executors promote
view tokens for the complete MLP call, and expired views fail validation instead
of dereferencing released storage. Device and pinned buffers retain the backend that
must free them. Scheduler queue/pending state is protected independently from the
residency map; event callbacks run after event-bus locks are released. A transfer
task's priority is immutable after dispatch, preventing worker/request races.
`ExpertManager` serializes its existing ownership transitions, while
`MemoryManager` independently protects accounting.

The transfer future is fulfilled only after `cudaEventSynchronize` through the
backend event API. A CUDA device allocation therefore cannot become visible to a
consumer while its `cudaMemcpyAsync` is incomplete. Transfer results include
bytes, wall-clock transfer duration, effective bandwidth, and whether CUDA handled
the movement. Scheduler subscribers receive CUDA-specific transfer events in
addition to tier-independent events.

## Deferred intentionally

- Native checkpoint readers and DeepSeek/GLM/Kimi/Mixtral adapters
- Batched or CUDA router execution, token generation, attention, and KV cache
- Direct-storage integrations and unbuffered platform-specific NVMe benchmarks
- Router-produced probabilities and measured compute/transfer overlap
- Automatic scheduler-driven capacity selection and eviction policy execution
- Quantized dequantization/GEMM, batched/strided GEMM, FP16 compute, kernel launch
  policy, and CUDA graphs
- Custom CUDA kernels
