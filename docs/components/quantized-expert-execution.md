# Quantized expert execution

Phase 22A adds a correctness-first INT8 expert path while preserving the same
storage, scheduling, residency, and backend boundaries used by floating-point
experts.

## Artifact representation

`ExpertPacker` accepts `ExpertPackingOptions`. The default `Preserve` mode is
unchanged. `Int8` mode transposes each gate, up, and down projection into the
runtime input/output layout, converts the source values to FP32 for calibration,
and applies deterministic symmetric per-tensor quantization:

```text
scale = max(abs(weight)) / 127
q = clamp(round(weight / scale), -127, 127)
weight ~= (q - zero_point) * scale
```

Zero tensors use scale `1`; non-finite input is rejected. The current writer
uses zero point `0`, while the runtime validates and executes any signed INT8
zero point. Each packed `ManifestTensor` stores the affine scale and zero point.
The expert index records INT8 dtype, the smaller byte range, and CRC32 for both
the complete expert and each projection. Conversion reports include source and
packed expert bytes, quantized projection count, and maximum observed per-weight
error. Existing floating-point artifacts remain readable and executable.

## Residency and execution

Quantized bytes are never expanded by the residency system:

```text
experts.bin INT8 -> RAM INT8 -> VRAM INT8 -> backend GEMM -> FP32 activation
```

`ExpertManager` still owns tier transitions and leases. `ExpertWeightMap` slices
the resident payload and binds scale metadata to each projection.
`ExpertMlpExecutor` dispatches FP32 weights to ordinary matmul and INT8 weights
to `matmulInt8Weights`; mixed storage is validated projection by projection.
The CPU backend is the scalar reference. Native CUDA builds use a device kernel
that reads INT8 weights and applies affine dequantization inside each FP32 dot
product. This kernel is deliberately simple and does not claim tensor-core or
production throughput.

## Numerical validation

For symmetric per-tensor quantization, an individual in-range weight has an
absolute reconstruction error no greater than approximately `scale / 2`.
Expert-output error also depends on input magnitude, reduction width, SiLU, and
the three projection scales; it is therefore measured rather than assigned a
universal `1e-5` tolerance. Phase 22 tests compare FP32 and INT8 expert outputs
with a fixture-specific bound and require CUDA INT8 GEMM to match the CPU INT8
reference within `1e-5` when native CUDA kernels are available.

## Benchmark scope

`hypermoe_quantized_expert_benchmark` emits
`hypermoe.phase22a-benchmark.v1`. It measures a deterministic CPU expert fixture
and a deterministic byte-capacity/LRU trace. It reports payload reduction,
transferred bytes, resident capacity, cache hit rate, first-execution latency,
throughput, and output error. These are fixture measurements, not RTX 4070 or
Qwen3-30B-A3B claims. Real checkpoint TTFT, tokens/sec, VRAM capacity, and cache
behavior must be collected by Hermes on the target CUDA machine.

One Release run on Apple Silicon (2026-09-14) produced the following fixture
measurements; timing values are machine-specific:

| Metric | FP32 | INT8 |
| --- | ---: | ---: |
| Expert payload | 393,216 bytes | 98,304 bytes |
| Experts fitting a 2 MiB comparison budget | 5 | 21 |
| Bytes transferred by the 2,000-request trace | 786,432,000 | 53,084,160 |
| Trace cache hit rate | 0.0% | 73.0% |
| First expert execution | 0.222 ms | 0.241 ms |
| Repeated single-token expert executions/sec | 6,808 | 9,993 |

The same run measured mean/max absolute output error of `0.000407`/`0.000919`.
The cache and transfer values are results of the benchmark's deterministic
synthetic trace, not projections for Qwen3.

## Limitations

- calibration is per projection, not per output channel or group;
- the CUDA kernel is an unfused scalar dot-product implementation;
- FP8, MXFP4, Q4 execution, batching, CUDA graphs, and transfer/compute overlap
  are outside Phase 22A;
- only expert MLP weights are quantized; attention, router, embeddings, norms,
  and LM head retain their existing storage paths;
- toolkit-only CUDA builds without a CUDA language compiler cannot execute the
  INT8 kernel and fail explicitly rather than staging hidden dequantization.
