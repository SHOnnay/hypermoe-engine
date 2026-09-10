# HyperMoE Engine

HyperMoE is a C++20 inference-runtime project for hierarchical Mixture-of-Experts
memory management across VRAM, pinned RAM, ordinary RAM, and NVMe. Phases 18 and
19 extend the optional CUDA path with device-resident activation, RMSNorm, RoPE,
router softmax/top-k, causal attention, and grouped expert gather/scatter. cuBLAS
continues to provide FP32 GEMM, while correctness-first CUDA kernels remove the
largest reference-path staging boundaries. CPU remains a complete fallback.
Real Qwen-compatible artifacts can be imported, validated, packed, and prepared
for traced CPU/CUDA comparison; no checkpoint or RTX benchmark is bundled or
fabricated. There is no server or chat API.

Phase 14.5 hardens the same runtime contracts across 64-bit little-endian
Windows, Linux, and macOS targets. It adds explicit wire-enum values, alignment
validation for external tensor storage, safer empty/moved buffer behavior,
portable index regression vectors, and CMake capability diagnostics. It does
not add inference features or alter scheduler policy.

Phases 15 and 16 add a bounded incremental generation foundation: tokenizer
adapters, session-owned KV caches, multi-token prefill, one-token decode,
greedy/temperature/top-k/top-p sampling, stop tokens, and text detokenization.
The real `ModelRuntime` is connected through a model-neutral generation adapter.
There is still no server, chat template, or built-in Qwen BPE implementation.

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
./build/hypermoe_cuda_runtime_benchmark cuda_runtime_report.json
./build/hypermoe_transformer_benchmark transformer_report.json
./build/hypermoe_model_runtime_benchmark model_runtime_report.json
./build/hypermoe_forward_benchmark forward_report.json
./build/hypermoe_real_model_benchmark /path/to/qwen-checkpoint \
  /path/to/packed-output real_model_report.json
```

Enable runtime memory checks with:

```sh
cmake -S . -B build-sanitize -DHYPERMOE_ENABLE_SANITIZERS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

CUDA runtime/cuBLAS support is detected with CMake's `CUDAToolkit` package;
native Phase 18 kernels are enabled only when a CUDA language compiler is also
available. A toolkit-only configuration retains the Phase 17 staged CUDA path.
When CUDA is absent, every target still builds and uses `CpuBackend`. To force a
CPU-only build:

```sh
cmake -S . -B build-cpu -DHYPERMOE_ENABLE_CUDA=OFF
```

On Windows, configure an x64 build from a Visual Studio developer shell:

```powershell
cmake -S . -B build -A x64 -DHYPERMOE_ENABLE_CUDA=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

CMake checks the C++20 standard library, 64-bit pointer width, 8-bit bytes, and
little-endian byte order before targets are created. `HYPERMOE_ENABLE_SANITIZERS`
enables ASan and UBSan with Apple Clang, Clang, or GCC; MSVC enables ASan and
prints that UBSan is unavailable. `HYPERMOE_WARNINGS_AS_ERRORS=ON` is available
for CI once every compiler's warning baseline is clean. See
[platform support](docs/components/platform-support.md).

The deterministic CPU generation benchmark writes a JSON report:

```sh
./build/hypermoe_generation_benchmark generation_report.json
./build/hypermoe_gpu_inference_benchmark gpu_inference_report.json
```

See [incremental generation](docs/components/generation.md), the
[tokenizer boundary](docs/components/tokenizer.md), and the
[KV cache runtime](docs/components/kv-cache-runtime.md). See
[CUDA inference](docs/components/cuda-inference.md) for the current accelerated
operations, staged correctness paths, backend selection, and validation rules.
The [GPU dataflow](docs/components/gpu-dataflow.md) and
[real-model validation](docs/components/real-model-validation.md) documents
describe the Phase 18/19 execution and artifact-validation boundaries.

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
reference implementations. CUDA uses cuBLAS GEMM and, with native-kernel support,
device activation and grouped gather/scatter; toolkit-only builds retain the
explicit staged fallback. No model-format assumption is included.

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

## NVIDIA execution validation

`CudaRuntimeValidator` verifies the selected device, compute capability, live
VRAM, CUDA runtime/driver versions, the three runtime streams, and event timing.
It emits a structured `PASSED`, `FAILED`, or `SKIPPED` report. A missing toolkit
or device therefore never produces invented GPU performance numbers.

The Phase 11 real-expert test imports and packs a format-correct Qwen3 MoE
SafeTensors artifact, routes a token, transfers the selected expert through
pinned memory into pooled VRAM, and executes its FP32 projections with cuBLAS.
The independent oracle compares router logits, top-k selection, projections,
activation, gating, and final output at `1e-5` tolerance. An active residency
lease prevents eviction until device synchronization completes.

FP16/BF16 CUDA storage currently uses an explicit selected-tensor conversion to
the FP32 correctness baseline. Native low-precision GEMM, quantized kernels, and
fused expert kernels remain deferred. See [CUDA runtime](docs/components/cuda-runtime.md),
[GPU execution](docs/components/gpu-execution.md), and
[hardware validation](docs/components/hardware-validation.md).

## Transformer execution pipeline

`TransformerBlock` composes four stable interfaces: `Attention`, `Norm`,
`MoELayer`, and `TensorBackend`. The Phase 12 reference order is attention,
RMSNorm, grouped top-k MoE execution, and a residual addition. `InferenceContext`
validates batch/layer identity and records routing and per-stage execution
metadata without carrying model-family logic.

`MoERuntime::executeBatch` routes all tokens together, requests every unique
expert once, gathers assigned token rows into an `ExpertBatch`, executes one MLP
per expert group, and scatters the routing-score-weighted results back into the
batch output. The existing single-token entry point delegates to this path.

The CPU attention implementation is deliberately single-head and non-causal; it
is a correctness foundation for Q/K/V projection, scaled scores, stable softmax,
context aggregation, and output projection. CUDA layer implementations can
replace the interfaces without adding CUDA branches to transformer code. See
[transformer runtime](docs/components/transformer-runtime.md),
[attention](docs/components/attention.md), and
[MoE layer](docs/components/moe-layer.md).

## Model-aware multi-layer runtime

The v2 manifest now optionally records validated runtime architecture metadata
and complete logical tensor bindings for each transformer layer. The Qwen
importer recognizes upstream Q/K/V/O, normalization, router, and expert tensors;
the offline packer rewrites them into neutral names and `INPUT_OUTPUT` layout.
`TransformerModelRuntime` consumes only those neutral mappings.

The CPU reference path supports multiple query heads, grouped key/value heads,
causal masking, RoPE at absolute sequence positions, per-layer KV storage, input
and post-attention RMSNorm, two residual branches, and grouped top-k MoE
execution across multiple layers. See [model runtime](docs/components/model-runtime.md),
[RoPE](docs/components/rope.md), and [KV cache](docs/components/kv-cache.md).

## Complete forward-to-logits runtime

`ManifestModelIO` binds token embeddings, the final normalization vector, and
the LM head without exposing source-model paths to execution code. Vocabulary
size and weight tying are explicit architecture properties. The packer preserves
the vocabulary-by-hidden embedding table and converts a separate LM head to the
backend's hidden-by-vocabulary layout. A tied head references the embedding
tensor directly, so its `TensorView` shares the same owner and bytes.

`ModelRuntime` composes embedding lookup, `TransformerModelRuntime`, final
RMSNorm, and vocabulary projection and returns logits plus measured timing for
each stage. Phase 14 executes this complete path on the CPU reference backend;
CUDA output components are intentionally deferred. See [model forward](docs/components/model-forward.md)
and [output head](docs/components/output-head.md).
