# Correctness validation

`CorrectnessOracle` is deliberately independent from `TensorBackend`, router
backend, and `ExpertMlpExecutor`. It uses scalar loops to calculate router logits,
stable softmax, deterministic top-k, SiLU-gated expert projections, and output
comparison. This separation makes offset, orientation, routing, and async
lifetime regressions observable instead of repeating the implementation under
test.

The Phase 9 fixture is a reduced but format-correct Qwen3 MoE artifact: a real
SafeTensors header and binary FP32 payload plus Qwen configuration metadata. The
test imports and packs it, validates projection index records and checksums,
loads one expert through the scheduler, creates zero-copy views under a residency
lease, runs the CPU executor, and compares output to the scalar oracle. Router
expert identity and normalized routing score are compared separately.

Default comparison tolerances are explicit metadata policy:

| Path | Absolute | Relative |
|---|---:|---:|
| FP32 | `1e-5` | `1e-5` |
| FP16 | `5e-3` | `5e-3` |
| BF16 | `1e-2` | `1e-2` |
| INT8 metadata | `2e-2` | `2e-2` |

Only FP32 execution is currently validated. The other entries define future
comparison policy; they do not claim implemented FP16/BF16/INT8 kernels.

Run `hypermoe_real_expert_benchmark` without arguments for the reduced fixture,
or pass a Qwen artifact and report path. The JSON report labels whether the
fixture was used and separates import, pack, load, tensor preparation, and
expert execution wall time. These are real local operations, unlike the existing
modeled pipeline benchmarks.
