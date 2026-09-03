# HyperMoE Engine

HyperMoE is a C++20 inference-runtime project for hierarchical Mixture-of-Experts
memory management across VRAM, pinned RAM, ordinary RAM, and NVMe. Phase 10 adds
Hugging Face multi-shard checkpoint validation, FP16/BF16-to-FP32 reference
execution, quantization policy metadata, and a minimal residual MoE transformer
layer over the real artifact path. There is intentionally no
complete transformer inference, tokenizer, server, or custom CUDA kernel.

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
./build/hypermoe_router_benchmark --tokens=100000 --report=router_report.json
./build/hypermoe_end_to_end_benchmark --tokens=10000 \
  --report=end_to_end_report.json
./build/hypermoe_model_inspect /path/to/model/metadata.json
./build/hypermoe_model_import /path/to/qwen-artifact hypermoe-manifest.json
./build/hypermoe_model_convert /path/to/qwen-artifact hypermoe_model
./build/hypermoe_real_expert_benchmark /path/to/qwen-artifact \
  real_expert_report.json
./build/hypermoe_model_validation_benchmark /path/to/qwen-artifact \
  model_validation_report.json
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
  manifest.json
```

`experts.index` retains the fixed 32-byte, little-endian expert record and v2 adds
fixed projection records for gate/up/down role, range, dtype, shape, and CRC32.
Payload offsets are 4 KiB aligned. The loader validates bounds and checksums and
reads only the requested expert or projection range. Legacy `metadata.json`
stores remain readable.

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
element strides, `FP32`/`FP16`/`BF16`/`INT8` dtype, CPU/CUDA device and ordinal, logical
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

## Model adapters and routing

`ModelAdapter` translates external metadata into `ModelMetadata`, capability
flags, `RouterConfig`, and `ExpertWeightMap`. The generic runtime uses only those
contracts; it never branches on `ModelArchitecture` or parses tensor names.
`QwenMoEAdapter` is the first validation adapter and contains all recognition of
Qwen-style `gate_proj`, `up_proj`, `down_proj`, and router tensor names.

The Phase 7 adapter input is the documented
`hypermoe.model-manifest.v1` JSON schema. This is an inspection/validation
manifest, not a claim that native Qwen checkpoints use this format. Phase 8's
Qwen SafeTensors importer now produces the more explicit v2 runtime manifest;
GGUF remains unsupported.

`CpuRouterBackend` calculates router logits for one hidden state, optionally
applies stable softmax, selects deterministic top-k experts, and optionally
renormalizes the selected scores. `MoERuntime` schedules all selected experts,
adopts completed device buffers into the layered `ExpertManager`, creates
zero-copy projection views, executes the gated MLP, and combines expert outputs.
No attention, transformer block, tokenizer, or token generation is included.

See [model adapters](docs/components/model-adapters.md),
[router](docs/components/router.md), and
[expert mapping](docs/components/expert-mapping.md) for the extension contracts.

## Real artifact import and prediction

`QwenImporter` reads an upstream Qwen2/Qwen3 MoE `config.json` and one or more
SafeTensors headers. It does not load tensor payloads. It validates dtype, shape,
file range, router metadata, and either per-expert projection tensors or Qwen3's
fused expert tensors, then writes `hypermoe.model-manifest.v2`. The manifest keeps
physical tensor locations separate from logical gate/up/down slices and records
the source matrix orientation. Runtime components consume the manifest rather
than interpreting the original checkpoint.

`Router::routeBatch` routes a rank-2 token batch and returns both per-token top-k
decisions and expert-grouped token indices. `TransitionDatabase` maintains
per-stream transitions, frequency, bounded recent selections, and expert
co-occurrence. `ExpertPredictor` turns those statistics into confidence-ranked
next-layer requests through the existing scheduler prefetch priority. Completed
prefetch buffers remain scheduler-owned until demand adopts them.

See [importer](docs/components/importer.md),
[manifest](docs/components/manifest.md), and
[predictor](docs/components/predictor.md). The end-to-end benchmark labels its
storage time as modeled; routing and prediction orchestration are measured CPU
work, not RTX 4070 results.

## Real artifact execution

`ExpertPacker` converts the imported v2 manifest into `manifest.json`, aligned
contiguous expert payloads in `experts.bin`, and a projection-aware
`experts.index`. `WeightConverter` explicitly normalizes matrix orientation to
`INPUT_OUTPUT`; names never imply layout. The runtime loads a whole expert once,
adopts the transfer buffer, and creates checked zero-copy projection views.

`ExpertResidencyLease` pins both logical residency and the physical buffer for
the complete executor call. Eviction excludes leased experts. The Phase 9 test
uses a format-correct reduced Qwen3 SafeTensors artifact and compares router
top-k, routing scores, and one FP32 expert output to an independent scalar oracle.
See [model conversion](docs/components/model-conversion.md),
[real execution](docs/components/real-execution.md), and
[correctness validation](docs/components/correctness-validation.md).

## Real-model compatibility and precision

`SafeTensorShardManager` validates `model.safetensors.index.json`, aggregates all
declared shards into one O(1) tensor lookup, and performs checked range reads from
the resolved file. `CheckpointValidator` compares imported manifest locations,
shapes, dtypes, expert mappings, and router mappings back to source metadata
without loading full tensors.

Packed FP16 and BF16 experts are converted projection-by-projection to FP32 by
the CPU reference runtime. `MoELayer` adds a validated identity attention
placeholder and residual connection around the existing router/scheduler/expert
path. Quantized INT8/Q4/Q8 policies are representable but not executable.

See [checkpoint validation](docs/components/checkpoint-validation.md),
[precision runtime](docs/components/precision-runtime.md), and
[transformer runtime](docs/components/transformer-runtime.md).
