# HyperMoE Engine

HyperMoE is a C++20 inference-runtime project for hierarchical Mixture-of-Experts
memory management across VRAM, pinned RAM, ordinary RAM, and NVMe. Phase 6 adds a
model-independent gated expert MLP above the CPU/optional CUDA tensor runtime,
with non-owning tensor views, checked INT8/Q4 descriptors, SiLU/GELU activation,
zero-copy resident-weight slicing, and execution-level profiling.
There is intentionally no transformer inference, model adapter, or custom CUDA
kernel implementation yet.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/hypermoe_simulation
./build/hypermoe_phase2_simulation --tokens=100000 --read-mode=mmap \
  --report=phase2_benchmark.json
./build/hypermoe_hardware_benchmark --report=hardware_report.json
./build/hypermoe_pipeline_benchmark --tokens=100000 \
  --report=pipeline_report.json
./build/hypermoe_cuda_benchmark --report=cuda_report.json
./build/hypermoe_tensor_benchmark --report=tensor_report.json
./build/hypermoe_expert_benchmark --report=expert_report.json
```

Enable runtime memory checks with:

```sh
cmake -S . -B build-sanitize -DHYPERMOE_ENABLE_SANITIZERS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

CUDA is detected with CMake's `CUDAToolkit` package. When it is absent, every
runtime target still builds and uses `CpuBackend`. To force a CPU-only build:

```sh
cmake -S . -B build-cpu -DHYPERMOE_ENABLE_CUDA=OFF
```

The Phase 1 simulator accepts `--requests`, `--seed`, `--vram-mib`, and
`--ram-mib`. The Phase 2 simulator accepts `--tokens`, `--seed`, `--read-mode`
(`mmap` or `range`), and `--report`. It creates a small temporary real expert
store, executes a seeded transformer-layer/top-k workload, then exports metrics.

## Expert store

A model directory contains:

```text
model/
  experts.bin
  experts.index
  metadata.json
```

`experts.index` has a versioned 32-byte header followed by fixed 32-byte,
little-endian records. Every record contains layer ID, expert ID, byte offset,
byte size, quantization code, and payload CRC32. Payload offsets are 4 KiB aligned.
The loader validates bounds and checksums and reads only the requested range.

See [architecture](docs/architecture.md) and
[design decisions](docs/design-decisions.md) for ownership and extension points.

## Hardware pipeline

`ComputeBackend` owns device allocation, copies, synchronization, memory queries,
streams, events, and pinned-memory primitives. `DeviceBuffer` and `PinnedBuffer`
provide automatic cleanup. `TransferManager` supports NVMe, RAM, pinned-RAM, and
VRAM sources/destinations, retains priority and cancellation behavior, invokes an
optional completion callback, and fulfills its future only after the backend event
completes.

With CUDA enabled, this maps to `cudaMalloc`, `cudaFree`, `cudaHostAlloc`,
`cudaMemcpyAsync`, nonblocking streams, and CUDA events. The CPU implementation
uses aligned allocations and `memmove`, making the same pipeline testable on
machines without NVIDIA hardware.

The hardware report distinguishes measured buffered filesystem throughput from
raw NVMe capability. Run it on the target RTX 4070 machine to obtain meaningful
pinned-memory and PCIe transfer numbers.

## Asynchronous scheduling

`Scheduler` is the consumer-facing coordination layer. It coalesces concurrent
requests for the same expert, upgrades queued speculative work when inference
needs it, and dispatches through a strict priority order: active inference,
predicted next layer, cache warming, then background maintenance. Its residency
state machine exposes `REQUESTED`, `QUEUED`, `LOADING`, `READY`, `IN_USE`,
`EVICTING`, and `FAILED`, including current/target tier, last-use time, and use
count. Consumers can wait on futures or subscribe to runtime events without being
coupled to storage and transfer implementations.

`LocalityPrefetcher` combines recent experts with a caller-provided next-layer
workload pattern. It is deliberately model-independent; a later router adapter
will provide real prediction signals. Prefetch hits, misses, queue wait, and
transfer overlap are exported by `Profiler`.

The pipeline benchmark is a deterministic event simulation. Its NVMe, RAM, and
GPU timing constants are inputs to a scheduling comparison and are not hardware
measurements. Use the hardware benchmark separately before tuning target-system
defaults.

## CUDA runtime foundation

`CudaRuntime` performs non-throwing device discovery and owns CUDA streams and
events when a usable device exists. `CudaStreamManager` creates three persistent
nonblocking streams: compute for future tensor execution, transfer for demanded
expert movement, and prefetch for speculative movement. The transfer manager uses
the latter two only for a CUDA backend; CPU builds contain no CUDA header or
runtime dependency.

`CudaMemoryPool` rounds allocations to 256-byte size classes, reuses the smallest
suitable cached block, and reports reserved, active, free, peak, allocation, and
reuse counters. `CudaBuffer` and pool-backed `DeviceBuffer` ownership return blocks
automatically. Pool shutdown frees cached blocks immediately and lets active
buffers safely release through retained control state later.

The CUDA benchmark measures direct and pooled VRAM allocation rate, pinned and
pageable host-to-device bandwidth, device-to-host bandwidth, and two-stream copy
overlap. When CUDA is not compiled or no NVIDIA device is usable, it still emits a
valid JSON report with zero GPU measurements and an explicit skip reason.

## Tensor and expert execution foundation

`Tensor` owns shared RAII storage and its metadata. It records shape,
element strides, `FP32`/`FP16`/`INT8` dtype, CPU/CUDA device and ordinal, logical
bytes, and backing-storage bytes. Shape and byte multiplication are checked for
overflow; strided tensors validate their full addressable span. Reshape shares
storage and currently requires contiguous layouts.

`TensorBackend` defines allocation, copy, matrix multiplication, elementwise add
and multiply, reshape, and synchronization. `CpuTensorBackend` supplies the
portable correctness implementation. `CudaTensorBackend` uses cuBLAS SGEMM for
contiguous rank-2 FP32 matrices and handles CPU↔CUDA and CUDA↔CUDA copies. CUDA and
cuBLAS remain optional CMake capabilities.

`TensorView` is a non-owning, lifetime-checked execution descriptor. It supports
read-only expert weights, writable outputs, checked reshape, and byte-offset
slices. `ExpertManager::residentDeviceTensorView` maps a transferred expert blob
directly onto its `DeviceBuffer`; the view does not increment ownership or copy
weight bytes. Scheduler residency must keep that buffer alive while it is in use.

`QuantizedTensor` owns or aliases checked packed storage and records shape,
device, positive scale, signed zero point, and `INT8` or packed signed `Q4` dtype.
Its versioned JSON metadata exposes the packed storage size. This phase does not
yet dequantize or execute quantized GEMM.

`ExpertMlpExecutor` performs the common gated primitive
`down(activation(input × gate) * (input × up))`. CPU SiLU and exact GELU are the
reference implementations. CUDA uses existing cuBLAS GEMM and an explicit
host-staged activation/elementwise fallback until profiling supports custom fused
kernels. No routing, transformer block, or model-format assumption is included.

The tensor benchmark reports CPU/CUDA FP32 GEMM, tensor allocation, copy, and
synchronization measurements. CUDA fields remain zero with an explicit skip
reason when cuBLAS or an NVIDIA device is unavailable.

The expert benchmark measures real expert-store loading, zero-copy tensor-view
preparation, projection GEMMs, activation, and total MLP execution. CPU and CUDA
results are reported separately, with an explicit CUDA skip reason when needed.
