# Design decisions through Phase 3

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

The current worker waits for its event before accepting another operation.
Multiple workers can overlap independent streams, but full inference/prefetch
overlap remains a later phase. Eviction will not free an allocation until all
consumers release its shared buffer. Custom kernels remain deferred until profiling
proves they are needed.

No direct-storage bandwidth assumptions are embedded. NVMe queue depth, range
size, mmap fault cost, pinned-memory behavior, and PCIe throughput must be measured
on the target RTX 4070 / i5-13400F system before tuning defaults.

## Hardware benchmark interpretation

CPU copy, RAM access, buffered sequential/random file reads, pinned-host copies,
and CUDA H2D/D2H copies are reported independently. Filesystem results may include
the OS page cache and are named `buffered` in JSON. They are regression signals,
not claims about raw SSD bandwidth. CUDA values remain zero with
`gpu_benchmarked: false` when no runtime device is available.
