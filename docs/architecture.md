# HyperMoE architecture through Phase 19

The runtime separates durable storage, movement, residency, eviction policy,
hardware access, and measurement so future model adapters do not own memory
policy.

```text
Qwen config + SafeTensors headers → ModelImporter
          │ validated physical locations + logical slices
          ▼
 HyperMoE v2 manifest → generic runtime metadata
          │
          ▼
 hidden-state batch + router weights → Router → grouped RouterDecision
          │ observations
          ▼
 ExpertHistory → TransitionDatabase → ExpertPredictor
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

Phase 9 adds an offline conversion branch before the store:

```text
Qwen SafeTensors → QwenImporter → source ModelManifest
                                   │
                                   ▼
                         WeightConverter + ExpertPacker
                                   │
                                   ▼
                    runtime manifest + experts.bin/index
                                   │
                                   ▼
 ExpertStore → Scheduler → ExpertManager → ExpertResidencyLease
                                   │
                                   ▼
                   TensorView gate/up/down slices
                                   │
                                   ▼
                         ExpertMlpExecutor
```

The Phase 10 model-compatible path adds shard and precision boundaries:

```text
HF shard index → SafeTensorShardManager → global tensor index
                         │
                         ▼
                 CheckpointValidator
                         │ exact manifest/source agreement
                         ▼
 packed FP32/FP16/BF16 expert → residency lease → TensorView
                         │ selected projections only
                         ▼
              DTypeConverter → FP32 reference execution
                         │
                         ▼
 identity attention placeholder → MoE runtime → residual output
```

Phase 11 validates the NVIDIA backend without changing model or scheduling
contracts:

```text
validated manifest + expert index
              │ active/prefetch request
              ▼
 Scheduler → TransferManager → PinnedBuffer → CUDA transfer stream
              │ completed event
              ▼
 CudaMemoryPool DeviceBuffer → ExpertResidencyLease → TensorView slices
              │
              ▼
 CudaTensorBackend → cuBLAS FP32 projections → synchronized output
              │
              ▼
 stage-by-stage CPU/CUDA CorrectnessOracle report
```

Phase 12 composes the first complete block-level reference pipeline:

```text
InferenceContext + hidden-state batch
                 │
                 ▼
       Attention interface → CPU Q/K/V + scaled softmax + output
                 │
                 ▼
              Norm interface → CPU RMSNorm
                 │
                 ▼
 batch Router → ExpertBatch groups → unique expert scheduling
                 │                         │
                 │                         ▼
                 └──────── weighted scatter-add ◄── grouped expert MLP
                                           │
                                           ▼
                         attention residual + MoE output
```

Phase 13 lifts that block into a manifest-defined model runtime:

```text
Qwen artifact → importer → architecture + logical layer mappings
                              │ offline layout conversion
                              ▼
                  packed neutral runtime manifest
                              │
                              ▼
             TransformerModelRuntime (layer 0 … layer N)
                              │
          input RMSNorm → causal MHA/GQA + RoPE ↔ KVCache
                              │ attention residual
                              ▼
          post-attention RMSNorm → grouped top-k MoE
                              │ MoE residual
                              ▼
                     next-layer hidden state
```

Phase 14 composes the complete forward-to-logits boundary:

```text
token IDs → manifest-mapped embedding table → hidden states
                                             │
                                             ▼
                              TransformerModelRuntime
                                             │ final hidden states
                                             ▼
                           manifest-configured final RMSNorm
                                             │
                                             ▼
                     separate LM head or shared embedding storage
                                             │
                                             ▼
                                  [tokens, vocabulary] logits
```

Phase 14.5 adds a portability boundary beneath those components rather than a
new execution stage:

```text
validated 64-bit little-endian platform contract
        │
        ├── fixed-width index fields; no struct serialization
        ├── checked filesystem ranges and moved mmap ownership
        ├── dtype-aligned tensor storage and lifetime-checked views
        ├── explicit enum wire values and invalid-value rejection
        └── optional CUDA/cuBLAS capability discovery
```

Phases 15 and 16 add stateful incremental execution above the forward runtime:

```text
Tokenizer -> InferenceSession -> Decoder::prefill
                                  |
                                  v
                         bounded KVCacheManager
                                  |
last logits -> LogitsProcessor -> Sampler -> Decoder::decode -> generated IDs
                                                               |
                                                               v
                                                       Tokenizer::decode
```

Phases 17 and 18 select a backend without changing model-facing contracts, then
keep the latency-critical sparse path resident on the selected device:

```text
hidden states (CUDA)
   |-- cuBLAS router logits -> CUDA softmax/top-k
   |                              |
   |                              v
   |                  CUDA grouped gather by expert
   |                              |
   |                    cuBLAS expert projections
   |                              |
   |                  CUDA activation + scatter-add
   |
   `-- cuBLAS Q/K/V -> CUDA RoPE -> contiguous CUDA KV cache
                                      |
                             causal score/mask/softmax/context
                                      |
                                cuBLAS output projection
```

The CMake CUDA capability boundary is two-stage: toolkit/runtime and cuBLAS can
be present without a usable CUDA compiler. In that configuration the Phase 17
staged path is retained; native kernels are compiled only when both capabilities
are available. CPU-only builds do not include CUDA headers or sources.

Phase 19 adds an artifact-validation boundary before performance claims:

```text
Qwen-compatible artifact -> QwenImporter -> CheckpointValidator
                                      | exact physical/logical agreement
                                      v
                                ExpertPacker
                                      |
                         packed runtime manifest/store
                                      |
               CPU trace <---- CorrectnessOracle ----> CUDA trace
                 logits + per-layer + per-expert intermediate tensors
```

Phase 20 connects that validated artifact to a reusable execution graph:

```text
HF SafeTensors shards + index + config + tokenizer metadata
                            |
                   QwenCheckpointLoader
                            |
                  RealCheckpointConverter
                            |
      manifest + experts.bin/index + tokenizer metadata/assets
                            |
                    PackedModelRuntime
               /                         \
 static tensors: one-time FP32 load       experts: scheduled on demand
               \                         /
       embedding -> transformer layers -> final norm -> LM head
                            |
              logits + attention/layer/expert trace
                            |
       RealModelValidator / RealModelProfileCollector
```

Storage dtype and execution dtype are distinct. Packed FP16/BF16 shared tensors
are promoted to FP32 while loading; expert projections are promoted only after
the selected expert becomes resident. The runtime never loads every expert into
RAM. CUDA and CPU construct the same graph from the same manifest.

## Components

- `ExpertIndex` parses a versioned, fixed-width little-endian format and builds
  an O(1) composite layer/expert lookup table.
- Index v2 appends projection descriptors and builds O(1) gate/up/down lookup;
  the original expert record remains byte-compatible with Phase 2 stores.
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
- `CudaTensorBackend` dispatches cuBLAS GEMMs and, when a CUDA compiler was
  detected, correctness-first kernels for activation, RMSNorm, RoPE, router
  selection, causal attention, and grouped expert gather/scatter.
- `CudaKVCache` grows geometrically as contiguous device storage. Attention can
  consume a logical device view directly; host snapshots remain available only
  for explicit oracle and fallback boundaries.
- `RealModelValidator` composes importer, source-checkpoint validation, expert
  packing, runtime-manifest validation, and deterministic CPU/CUDA trace
  comparison without embedding Qwen tensor names in the runtime.
- `QwenCheckpointLoader` validates config, sharded SafeTensors/index mappings,
  complete forward tensors, tokenizer vocabulary IDs, special-token IDs, and
  tokenizer metadata before conversion starts.
- `RealCheckpointConverter` transactionally creates a self-contained runtime
  artifact and preserves the original artifact if an overwrite is attempted.
- `PackedModelRuntime` binds shared tensors once, keeps experts under scheduler
  residency, selects CPU or CUDA components, and executes single-sequence token
  IDs to logits.
- `RealModelProfileCollector` reports measured load/forward/decode time,
  layer-qualified expert frequency, transfers, cache/prefetch counters, KV
  storage, and runtime-accounted RAM/VRAM.
- `CudaRuntime` initializes a selected device, reports compute capability and
  live VRAM information, owns created streams/events, and shuts them down after
  synchronization. With CUDA disabled it remains queryable and reports
  unavailable without including CUDA headers in public interfaces.
- `CudaDeviceInfo` provides the explicit CUDA hardware record while preserving
  the earlier `DeviceInfo` API alias. `CudaRuntimeValidator` checks properties,
  VRAM, versions, three streams, and timed events and exports a structured report.
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
- `ExpertResidencyLease` pins a resident expert across view creation, executor
  work, and backend synchronization. Leased experts are not eviction candidates.
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
  behavior. CUDA projection uses cuBLAS; native-kernel builds keep activation,
  grouped gather, and scatter-add on device.
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
- `MoERuntime` coordinates batch route → parallel unique-expert scheduling →
  buffer adoption → grouped expert execution → routing-weighted scatter-add.
  The single-token API delegates to this path. Its current execution lock
  deliberately serializes layer calls while the contracts mature.
- `ExpertHistory` records per-layer expert frequency, cross-layer transitions,
  and the previous selection for profiling and predictor integration.
- `SafeTensors` inspects bounded JSON headers and validates tensor ranges against
  every shard without reading weight payloads. BF16 storage is representable and
  selected CPU projections can now be expanded to FP32 reference execution.
- `SafeTensorShardManager` validates Hugging Face weight maps, aggregates shard
  metadata, and resolves checked tensor-relative range reads.
- `CheckpointValidator` proves that imported tensor locations and metadata still
  match the source checkpoint without reading full payloads.
- `DTypeConverter` selects an explicit execution plan and expands selected
  FP16/BF16 weights to FP32 on CPU or through a CUDA host-staging path. Storage
  dtype remains unchanged in the packed model.
- `QuantizationPolicy` records validated INT8/Q4/Q8 scale, zero-point, and group
  metadata without implying an optimized kernel.
- Phase 22A's deterministic INT8 packer stores one affine scale/zero point per
  expert projection in manifest v3. The index continues to checksum exact
  projection and expert byte ranges, so INT8 uses the same random-access and
  corruption-detection path as floating-point storage.
- `Attention` and `Norm` are device-neutral layer contracts. `CpuAttention`
  supplies FP32 multi-head/grouped-query Q/K/V, causal scaled softmax, context,
  and output projection; `RMSNorm` supplies the CPU reference normalization.
- `ExpertBatch` represents all token rows and routing weights assigned to one
  expert. `MoELayer::executeExpertsBatch` exposes grouped execution while
  retaining the legacy single-token and identity-wrapper APIs.
- `InferenceContext` validates batch/layer dimensions and records routing plus
  execution metadata. `TransformerBlock` composes attention, RMSNorm, grouped
  MoE, and residual addition without backend-specific branches.
- `models::runtime::ModelArchitecture` validates manifest-derived attention,
  normalization, routing, expert, and layer dimensions. `ManifestLayerMapping`
  binds execution roles to neutral tensor names and layouts.
- `TransformerModelRuntime` resolves owned shared tensors once and executes every
  mapped block in layer order while retaining per-layer timings, routing, and
  correctness outputs.
- `Embedding` performs checked CPU FP32 row lookup from a manifest-bound
  vocabulary-by-hidden table. Token IDs never imply a tokenizer or text format.
- `FinalNorm` applies the architecture's final RMSNorm epsilon and validated
  hidden-width scale tensor independently of per-layer normalization.
- `LMHead` projects final hidden states to vocabulary logits. It accepts the
  packed hidden-by-vocabulary layout or the vocabulary-by-hidden layout required
  to alias tied embedding storage.
- `ModelRuntime` owns no duplicate weights: it resolves non-owning views from the
  `RuntimeTensorMap`, composes the four forward stages, and reports stage timings.
- `CpuAttention` supports causal multi-head and grouped-query attention. `RoPE`
  rotates query/key pairs at absolute positions before the per-layer `KVCache`
  stores keys and values.
- `ModelImporter` is the architecture-independent artifact boundary.
  `QwenImporter` alone interprets Qwen configuration keys and tensor names,
  including individual Qwen2 projections and fused Qwen3 expert storage.
- `ModelManifest` v2 records source files, physical tensor byte ranges, router
  configuration, output/input matrix orientation, and per-expert gate/up/down
  slices. Runtime readers validate all references before use.
- `ExpertPacker` range-reads those slices, explicitly transposes declared layouts,
  and creates the three-file runtime store without interpreting Qwen names.
- `CorrectnessOracle` independently computes router and gated-MLP references,
  selects top-k from externally computed logits, and reports CPU/CUDA numerical
  differences for every execution stage.
- Batch routing returns one decision per token plus groups of token indices and
  scores for each selected expert, preparing grouped expert dispatch.
- `TransitionDatabase` keeps statistics per inference stream so tokens in a batch
  do not create false transitions between independent sequences.
- `ExpertPredictor` uses transition probability, frequency, recency, and
  co-occurrence. It emits ordinary predicted-next-layer scheduler requests; it is
  statistical and does not claim model-level learned routing.
- Phase 21 predictions separate normalized next-layer probability from evidence
  confidence. Lazy-decayed transitions, a bounded popularity window, and
  per-stream evaluation feed confidence-gated prefetch and deterministic
  same-priority scheduler ordering.
- `Scheduler` retains completed prefetch transfer buffers. A subsequent demand
  request receives the exact buffer and storage record instead of observing a
  READY state whose data ownership has expired.
- `CachePolicy` has interchangeable LRU, LFU, and hybrid implementations. The
  hybrid normalizes signals before applying 0.4 frequency, 0.3 recency,
  0.2 layer probability, and 0.1 prefetch confidence.
- `ExpertManager` records those same score inputs per expert and exposes a
  residency snapshot. Packed runtimes select hybrid eviction by default while
  retaining LRU through configuration for regression and benchmark baselines.
- `MemoryPressureController` restores configurable safety margins and monitors
  transfer-queue depth.
- `Profiler` collects requests, transfers, evictions, pressure, CUDA/NVMe/RAM
  timing, GPU-memory use, queue depth, byte counts, stalls, prefetch outcomes,
  average scheduler queue wait, transfer overlap, kernel/matmul/expert/projection/
  activation/quantization time, tensor allocations, and externally supplied
  GPU-utilization observations.
- `RuntimeMetricsSnapshot` projects memory, MoE, prediction, cache, and execution
  counters into a stable read-only in-process interface. No presentation or
  transport layer is coupled to runtime ownership.
- `HardwareInfo` reports CPU, logical cores, RAM, available storage, CUDA
  build/runtime state, GPU name, VRAM, runtime version, and driver version.
- The CMake platform gate requires C++20 library support, a 64-bit target,
  8-bit bytes, and little-endian byte order. These are explicit packed-artifact
  compatibility requirements rather than implicit host assumptions.
- Tensor owners and views reject storage that is not naturally aligned for its
  dtype. CPU GEMM rejects input/output aliasing exactly as the CUDA backend does.
- The Windows mapped-file implementation uses `nullptr` as its internal closed
  state; moved and closed mappings expose empty spans without sentinel pointer
  arithmetic.
- `KVCacheManager` performs conservative per-session admission against a
  configured aggregate memory limit while `KVCache` grows dynamically.
- `InferenceSession` owns sequence, forward, and cache state. `Decoder` executes
  multi-token prefill and one-token cached forwards without changing the
  stateless Phase 14 API.
- `GenerationModel` decouples orchestration from model internals;
  `ModelRuntimeGenerationModel` supplies the real forward-to-logits bridge.
- `Tokenizer`, `LogitsProcessor`, `Sampler`, and `Generator` keep text mapping,
  probability policy, and decode control outside transformer execution.
- `CudaTensorBackend` keeps FP32 GEMM and device copies on CUDA and now implements
  residual addition with cuBLAS AXPY plus elementwise multiplication with cuBLAS
  diagonal scaling. Optional kernels add activation, RMSNorm, RoPE, top-k,
  causal attention, and sparse gather/scatter. The CPU backend remains the
  numerical reference.
- `TensorBackend::matmulInt8Weights` is the quantized expert seam. CPU executes
  the scalar affine reference; native CUDA reads INT8 weights and dequantizes
  them inside FP32 dot products without materializing a second resident weight
  tensor.
- `CudaAttention` executes QKV and output projections through the CUDA tensor
  backend. Native builds also execute RoPE, causal score/mask, stable softmax,
  and context accumulation in CUDA; toolkit-only builds use the reference seam.
- `CudaRouterBackend` keeps logits, normalization, and top-k selection on-device
  when native kernels are available. CUDA-capable `Embedding` and `LMHead`
  retain checked host-staged fallbacks.
- `KVCacheBase` separates the sequence contract from storage. `KVCache` owns host
  arrays and `CudaKVCache` owns geometrically growing contiguous device tensors;
  `KVCacheManager` selects the implementation from its configured tensor backend.
- `InferenceConfig` binds a session to an explicit device. The model, cache
  manager, and requested device must agree before cache allocation or execution.
- Generation materializes only the logits needed by the sampler through the
  model adapter, preserving a single API for CPU and CUDA execution.

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

## Phase 22B transfer overlap and budgets

Phase 22B removes the all-selected-experts transfer barrier in `MoERuntime`.
Execution waits for one expert, adopts its completed buffer, holds a residency
lease, prepares room for one lookahead expert, and submits that expert before
executing the current one. If two payloads cannot fit, loading remains serial.
The transfer manager owns transfer/prefetch streams and completion events;
the tensor backend owns the compute stream. No device-wide wait is added.
The transfer event is the readiness boundary; compute-stream completion is the
lease-release boundary. Existing per-operation profiling waits remain.

Packed-runtime statistical prefetch stages into RAM, with bounded speculative
scheduler ownership, rather than independently accumulating VRAM buffers.
Completed active results transfer ownership to `ExpertManager`; consumed
futures and scheduler cache entries release their references. Scheduler metadata
is reconciled against manager residency before requests after eviction.
The expert budgets govern manager residency; pinned buffers, bounded speculative
warm cache, pool alignment/free blocks, static tensors, KV cache and execution
temporaries are separate physical overhead. See
[expert overlap](components/expert-overlap.md) for the measurement contract.

## Phase 22C: safe expert capacity and aging

Auto CUDA sizing runs after static loading and uses the tighter of sampled free
VRAM and total-minus-static capacity. KV, workspace, staging/free-pool, safety
and transfer/alignment reservations are removed before creating the existing
`MemoryManager` limit. Manual defaults remain unchanged. Auto cache capacities
and forward dimensions are checked against envelopes; this is startup sizing,
not elastic arbitration of global GPU memory.

Packed adaptive residency selects age-aware Hybrid: decayed frequency/recency
and confidence-weighted, aging predictor hints determine eviction. Active leases
remain protected. Diagnostics expose the actual score and successful device
promotions. No scheduler/transfer/attention/INT8 architecture is replaced. See
[adaptive residency](components/adaptive-residency.md).

## Deferred intentionally

- GGUF readers and DeepSeek/GLM/Kimi/Mixtral artifact importers
- CUDA embedding lookup, LM head specialization, and paged KV cache layout
- Direct-storage integrations and unbuffered platform-specific NVMe benchmarks
- Executing tokenizer vocabulary/merges or chat templates, Qwen2 shared-expert
  residuals, dense/non-MoE layers, output bias, batched sessions,
  beam/speculative decoding, streaming, and serving
- Automatic scheduler-driven capacity selection and eviction policy execution
- Optimized/tensor-core quantized GEMM, batched/strided GEMM, FP16 compute,
  kernel launch policy, and CUDA graphs
- Fused or precision-specialized CUDA kernels
