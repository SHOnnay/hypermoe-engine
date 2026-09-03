# Design decisions through Phase 8

## Why memory mapping

Expert models are much larger than available RAM. A read-only mapping lets the OS
page cache serve random expert regions without an application-sized copy or a
long-lived file cursor. HyperMoE still copies only the selected region into an
owning staging buffer because residency must remain valid independently of the
mapping. Explicit range reads remain available for measurement and platforms or
workloads where mmap page-fault behavior is undesirable.

## Why a binary index

Routing needs constant-time lookup and predictable parsing. The index uses an
explicit little-endian header and fixed-width records instead of serializing C++
struct memory, avoiding ABI padding and endianness ambiguity. A version, endian
marker, record width, and count reject incompatible files early. Layer/expert
keys are checked for duplicates; offsets and sizes are checked against
`experts.bin`; CRC32 detects truncated or corrupted payloads. Four-KiB payload
alignment supports efficient range I/O without claiming any specific SSD's
minimum transfer behavior.

CRC32 is for accidental corruption detection, not authenticity. A future model
manifest may add a cryptographic digest or signature without changing records.

## Why the cache hierarchy exists

Consumer VRAM cannot hold all sparse experts, while RAM is faster and lower
latency than NVMe. Exclusive logical residency makes ownership and limits clear:
hot experts occupy VRAM, useful demotions occupy RAM, and the immutable complete
copy remains in storage. Policy selection is separate from movement. LRU is a
baseline, LFU captures stable popularity, and hybrid scoring accepts routing and
prefetch signals. Signal normalization prevents frequency counts from drowning
probabilities.

Safety margins are explicit rather than waiting for allocation failure. The
pressure controller can demote before backend work needs emergency capacity and
also reports transfer-queue pressure.

## Why a worker queue and futures

`std::async` provides a convenient loader API but does not bound concurrency or
prioritize work. `TransferManager` owns a fixed worker pool and stable FIFO order
within each priority. Futures propagate storage errors without callbacks crossing
ownership boundaries. Cancellation is cooperative: queued work cancels promptly;
an in-progress blocking filesystem read finishes before its result is discarded.

## Optional hardware backend

CUDA cannot be a build or deployment requirement for storage tooling and CPU-only
tests. The abstract backend therefore exposes allocation, copies, synchronization,
memory information, pinned allocation, streams, and events. CUDA translation code
is guarded by `HYPERMOE_HAS_CUDA`; CMake defines it only after finding CUDAToolkit.
Runtime device detection remains separate because a machine can have toolkit
headers but no usable GPU or driver. The CPU backend models device ownership with
aligned memory; it is a correctness fallback, not a CUDA performance prediction.

## Pinned-memory staging

Page-locked memory avoids pageable-memory staging performed internally by the CUDA
runtime and enables asynchronous PCIe copies. `PinnedBuffer` attempts
`cudaHostAlloc` only through a capable backend and otherwise uses a 64-byte-aligned
allocation. Its RAII lifetime prevents page-locked leaks, which are particularly
harmful because excessive pinned memory reduces memory available to the OS.

## CUDA transfer lifetime

The CUDA backend allocates with `cudaMalloc`, queues `cudaMemcpyAsync` on a
nonblocking stream, records an event, waits for completion, and only then publishes
the `DeviceBuffer` through the future. CUDA timing events accumulate copy duration
when the stream is synchronized. Device and pinned buffers retain shared backend
ownership, so the backend cannot disappear before cleanup.

The current transfer worker waits for its event before accepting another
operation. Multiple workers can overlap independent streams, while the scheduler
can place predicted work ahead of maintenance and behind demand. Eviction will not
free an allocation until all consumers release its shared buffer. Custom kernels
remain deferred until profiling proves they are needed.

No direct-storage bandwidth assumptions are embedded. NVMe queue depth, range
size, mmap fault cost, pinned-memory behavior, and PCIe throughput must be measured
on the target RTX 4070 / i5-13400F system before tuning defaults.

## Hardware benchmark interpretation

CPU copy, RAM access, buffered sequential/random file reads, pinned-host copies,
and CUDA H2D/D2H copies are reported independently. Filesystem results may include
the OS page cache and are named `buffered` in JSON. They are regression signals,
not claims about raw SSD bandwidth. CUDA values remain zero with
`gpu_benchmarked: false` when no runtime device is available.

## Why asynchronous expert scheduling is required

Sparse inference reduces arithmetic but makes weight arrival part of the critical
path. A synchronous caller pays NVMe, host staging, and PCIe latency before every
cold expert execution. A scheduler can start likely next-layer movements while
the current expert computes, then expose a future only when the destination data
is safe to consume. The useful optimization target is therefore hidden transfer
time and reduced stalls, not nominal SSD bandwidth.

The scheduler is separate from `TransferManager`: transfers know how bytes move,
whereas scheduling knows why an expert is needed, whether another consumer already
requested it, and which request may delay inference. Duplicate expert/target
requests share a future. A queued prefetch is upgraded when demand arrives; a
transfer already in progress is not mutated because backend priority changes
cannot reliably preempt issued I/O.

## Why an explicit residency state machine

Tier location alone cannot distinguish data that is usable from data whose copy
is queued or incomplete. The lifecycle makes availability auditable and rejects
unsafe transitions. `current_location` remains the last valid tier while
`target_location` describes in-flight movement. `IN_USE` blocks eviction at the
state boundary, and `FAILED` provides a recoverable state for cancellation,
checksum errors, allocation errors, or backend failures.

## Priority, events, and prediction

Priority order is active inference, predicted next layer, cache warming, and
background maintenance. FIFO sequence numbers retain deterministic ordering
within a class. This prevents speculative work from extending a demand stall while
still allowing idle bandwidth to warm the hierarchy.

Runtime events keep cache bookkeeping, metrics, and future API consumers out of
the scheduler's transfer logic. Callbacks execute without holding event-bus locks;
observer exceptions are isolated. Futures remain the authoritative completion and
error channel.

Prefetch prediction is an interface rather than embedded router logic. The
baseline locality predictor scores recent experts and caller-provided next-layer
patterns. It is useful for testing scheduling mechanics, but its confidence is not
a claim about Qwen, GLM, DeepSeek, or Kimi routing. Real adapters remain deferred
until their metadata and router behavior are inspected.

## Pipeline benchmark interpretation

The synchronous/async benchmark is a deterministic discrete-event simulation. It
uses explicit modeled compute, NVMe, and RAM times so scheduling changes can be
regression-tested without sleeps or a GPU. Reported latency, stalls, and hidden
transfer percentage are comparative model outputs—not measured tokens/second,
PCIe throughput, or SSD performance. Hardware defaults must be derived from the
separate hardware benchmark on the target RTX 4070 system.

## CUDA runtime ownership

CUDA handles do not appear in public headers; `StreamHandle` and `EventHandle`
remain opaque. `CudaRuntime` is the single RAII owner for handles it creates and
sets the selected device on every calling thread before device-specific work.
This matters because transfer workers are not the thread that initialized the
backend. Discovery returns an unavailable `DeviceInfo` rather than throwing, so
CPU-only executables and benchmark reporting remain straightforward. Constructing
an actual `CudaBackend` still fails clearly when no device is usable.

## Why a device memory pool

Expert churn would otherwise put `cudaMalloc` and `cudaFree` on the routing hot
path. The pool retains released blocks and performs best-fit reuse after rounding
to 256-byte capacities. Logical size remains separate from capacity, preventing a
consumer from treating padding as tensor data. The default cache bound is
configurable and cached blocks can be trimmed under future memory pressure.

Pool statistics distinguish reserved bytes (active plus cached), active bytes,
free cached bytes, peak active use, physical allocation count, and reuse count.
An internal shared control state is captured by both `CudaBuffer` and adopted
`DeviceBuffer` releasers. If the facade shuts down while a buffer is active, the
cache stops accepting returns and that buffer frees directly when released.

## Stream roles and copy completion

Demand transfers and speculative prefetches use separate persistent nonblocking
streams; a third stream is reserved for future expert computation. This creates
the synchronization boundary needed to overlap future kernels with weight copies
without implying that the current phase launches kernels. Transfer completion is
event-based, and futures are published only after the event is synchronized.

Wall-clock duration and effective bandwidth are reported per transfer. CUDA event
timings remain accumulated by the backend for device-copy profiling. The CUDA
benchmark compares pinned/pageable transfers and serial/multiple-stream copies,
but actual overlap depends on the GPU copy engines, PCIe topology, driver, buffer
size, and system load.

## Tensor execution boundary

Tensor storage is shared separately from tensor metadata. This allows reshape to
create a zero-copy view and lets an expert's pool-backed `DeviceBuffer` become a
tensor without changing ownership. Shapes validate both logical element count and
the full addressable span implied by strides. Operations currently require
contiguous layouts, making unsupported views fail before pointer arithmetic.

The tensor backend is distinct from `ComputeBackend`: the latter owns bytes,
streams, and copies, while the former validates mathematical shape/dtype contracts
and selects an implementation. CPU operations are intentionally simple reference
loops. They prioritize deterministic correctness and sanitizer visibility over
vectorization.

## Why cuBLAS comes before custom kernels

FP32 GEMM is the first execution primitive because an expert projection reduces
to matrix multiplication. cuBLAS already provides architecture-tuned GEMM,
well-defined stream behavior, and mature correctness on NVIDIA GPUs. HyperMoE
adapts row-major tensors to cuBLAS's column-major SGEMM convention by computing
the transposed result layout without inserting transpose copies.

Writing a custom GEMM before profiling would duplicate a highly optimized library
and expand the correctness surface. Custom kernels remain candidates for fused
bias/activation/gating, dequantization, or layouts that measurement shows cuBLAS
cannot handle efficiently. CUDA elementwise add/multiply currently use an explicit
host-staged reference path rather than pretending that this fallback is optimized.

## Why tensor ownership and use are separate

An owning `Tensor` is appropriate for activations and temporary results, but it is
the wrong default descriptor for cache-managed expert weights: copying shared
ownership into every operation would silently extend residency and prevent the
pressure controller from reclaiming cold experts. `TensorView` therefore copies
only shape/dtype/device metadata and keeps a weak lifetime token. It can expose a
whole `DeviceBuffer` or checked byte slices without a weight allocation or copy.
Backends temporarily promote that token while an operation is active; the MLP
executor pins all participating views across its complete operation sequence.

This makes the lifetime contract explicit. Scheduling must keep an expert in
`IN_USE` while views execute. If the owner disappears first, the view becomes
invalid and backend validation rejects it. The weak token does not make an
unsynchronized concurrent eviction safe by itself; scheduler pinning is the
required synchronization boundary.

## Expert execution architecture

The generic executor implements the gated MLP used by many sparse experts:
`down(activation(input × gate) * (input × up))`. Gate and up projections share
input/model dimensions, while down maps the intermediate width to output width.
All contracts are validated before allocating intermediate activations.

SiLU and exact-erf GELU are CPU reference functions. The same activation dispatch
accepts a CUDA tensor backend, but currently stages through CPU memory. Projection
GEMMs continue to use cuBLAS on CUDA. This provides end-to-end correctness and a
stable activation seam without introducing an unprofiled kernel. The benchmark
reports loading, view preparation, projection, activation, and total execution so
the host-staging cost will be visible on NVIDIA hardware.

## Quantization roadmap

`QuantizedTensor` separates packed dtype from ordinary `Tensor::DType`. INT8 uses
one byte per value; signed Q4 packs two values per byte. Both require contiguous
storage and carry a positive finite scale plus an in-range signed zero point.
Versioned serialization metadata records shape, device, parameters, and packed
byte size, while `ExpertManager` checks the descriptor against storage metadata.

Phase 6 deliberately does not guess a block/group quantization layout or execute
quantized GEMM. Real model metadata must define group size, scale tensor layout,
packing order, and any per-channel rules before dequantization or fused kernels
are added. INT8/Q4 execution, FP16 accumulation, and fused gated kernels remain
future profiling-driven work.

## Why the core is adapter-driven

Model families disagree on tensor names, expert layout, shared experts, routing
normalization, and quantization metadata. Encoding any one naming convention in
the scheduler or executor would turn future integration into conditionals spread
across memory and compute code. An adapter therefore produces capabilities and a
neutral metadata graph. Core execution consumes generic `GATE`, `UP`, and `DOWN`
roles and never examines an architecture enum or source tensor name.

`ModelArchitecture` is diagnostic information for tools and adapter selection.
Capabilities are the behavioral contract. This lets a future adapter describe a
feature combination without requiring runtime logic to infer behavior from a
family label.

## Why the first adapter uses a neutral manifest

Phase 7 had no native checkpoint parser and did not invent one. The Qwen-style
adapter accepts `hypermoe.model-manifest.v1`, a strict JSON inspection schema with
explicit shapes, dtypes, offsets, sizes, layer/expert IDs, router settings, and
tensor names. It cross-checks Qwen-style names against explicit IDs and validates
complete projection sets. The schema is an adapter validation input, not a claim
about GGUF, SafeTensors, or upstream Qwen files.

The JSON parser preserves unsigned 64-bit offsets as number text until checked,
rejects duplicate keys and malformed Unicode, limits file size and nesting, and
reports byte positions. Native readers can later construct the same
`ModelMetadata` without changing routing or execution.

## Router and top-k semantics

The reference router computes `hidden × router_weights`, rejects non-finite
scores, applies stable softmax when configured, and selects any `k` in
`[1, expert_count]`. Equal scores use ascending expert ID, making tests and
benchmarks deterministic. Selected scores may be renormalized independently;
this is configuration rather than a model-family assumption.

The CPU implementation retains the one-token entry point and now also routes a
rank-2 token batch. A future CUDA backend can implement the same `RouterBackend`
contract without changing scheduler or model code. Capacity factors, expert
masks, token dropping, and auxiliary-loss behavior remain outside this phase.

## Layered identity and scheduler adoption

Expert IDs are normally local to a transformer layer. `ExpertManager` now keys
ownership by `(layer_id, expert_id)` and assigns an internal cache-policy ID.
Legacy single-ID calls resolve only unique IDs and fail clearly when ambiguous.

The scheduler already transfers by composite identity. After a successful load,
`MoERuntime` adopts the returned `DeviceBuffer` into `ExpertManager` with logical
memory accounting; it does not issue a second disk read. Scheduler `acquire` and
`release` bracket the zero-copy views used by `ExpertMlpExecutor`.

## Prediction preparation

`ExpertHistory` records selection frequency, transitions from every prior
selection to every current selection, and the most recent decision. The router
benchmark exercises this data with a deterministic 100,000-token workload. No
prediction policy is claimed yet: the next predictor can consume history and
adapter/router signals through a separate interface and submit ordinary
prefetch-priority scheduler requests.

## Why import produces a second manifest

Source checkpoints are distribution formats, not runtime contracts. Their tensor
names, sharding, physical orientation, fused projections, and configuration keys
can change independently from HyperMoE scheduling. Phase 8 therefore inspects a
source artifact once and emits a strict v2 manifest. Runtime code consumes only
validated physical locations and logical expert roles.

Phase 8 import alone does not copy or convert payloads: SafeTensors inspection
reads the bounded header, validates every declared range against its shard, and
records relative file paths. Phase 9's separate offline packer then uses that
manifest to build `experts.bin` without parsing Qwen names again. Absolute paths
and parent-directory traversal are forbidden so a manifest remains relocatable
and cannot escape its artifact root.

## Why Qwen supports two expert layouts

Older Qwen MoE artifacts can expose separate gate/up/down tensors per expert,
while current Qwen3 MoE implementations can store a three-dimensional fused
`gate_up_proj` and a fused `down_proj`. The importer represents both as the same
logical expert mapping. Slices retain `OUTPUT_INPUT` orientation because source
linear weights are not silently transposed during metadata import.

Supporting BF16 in metadata does not imply BF16 execution. The distinction lets
inspection describe real artifacts accurately while backends continue to reject
unsupported math paths.

## Why prediction is statistical first

Transition probability is cheap, explainable, deterministic, and can be updated
online without a training system. Frequency prevents sparse observations from
overweighting a single edge, recency preserves current locality, and
co-occurrence captures top-k expert pairs. Statistics are keyed by inference
stream so a batch does not create transitions between unrelated sequences.

Predictions use the scheduler's existing lower priority and cancellation model.
They do not bypass residency or transfer validation. The scheduler retains a
completed speculative buffer until a demand request consumes it, which makes
READY mean that bytes are still owned—not merely that a past future completed.

## Batch routing boundary

The router backend owns batched score calculation and top-k selection. Its result
contains both per-token decisions and deterministic expert groups. This avoids
forcing the future executor to repeatedly scan all token decisions when building
an expert batch. Capacity factors, token dropping, and grouped GEMM remain
deferred because they are model/runtime policies beyond reference routing.

## Why conversion is offline and manifest-driven

SafeTensors is optimized for distribution and inspection, not repeated
expert-granular residency movement. Offline packing makes each expert one aligned
contiguous range while preserving projection subranges and checksums. The packer
consumes the importer manifest rather than parsing Qwen names, so format-specific
knowledge remains at the artifact boundary and the runtime contract is stable.

Layout conversion is explicit. Upstream linear weights are commonly stored as
`OUTPUT_INPUT`, while the current row-major backend consumes `INPUT_OUTPUT`.
Transposing once during packing removes hidden runtime copies and makes a wrong
orientation detectable through shape and oracle tests.

## Why execution uses a residency lease

A `TensorView` intentionally does not own expert bytes. Backend-level temporary
promotion prevents deallocation during one operation, but an expert MLP is a
sequence of operations and the cache can otherwise move the payload between
them. A move-only residency lease increments a manager pin count and owns the
device buffer for the whole executor call. Eviction selection ignores pinned
experts and explicit movement rejects an active lease. This is the synchronization
boundary for future asynchronous CUDA work as well as current CPU execution.

## Why the correctness oracle is independent

Reusing tensor matmul or router code in expected-value generation would allow the
same layout or offset defect to pass twice. The oracle therefore uses scalar
loops and its own routing/top-k implementation. FP32 is the executed Phase 9
path; FP16, BF16, and INT8 tolerances are recorded now as validation policy, not
as claims that those execution paths exist.
