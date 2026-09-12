# Runtime optimization and metrics

Phase 21 connects router observations to prediction, prefetch, scheduling, and
residency without changing model formats or the CPU/CUDA backend boundary.

Prefetch requests remain below active inference in the scheduler's primary
priority bands. Within the predicted-next-layer band, a deterministic score
combines probability and confidence. This lets a demanded expert always
preempt speculative work while placing the strongest speculation first.
Completed prefetches that are consumed are recorded as useful; late prefetches,
low-confidence suppressions, and completed entries that age past their layer
are reported independently.

The adaptive residency path uses the existing `HybridPolicy` contract. Each
expert has usage count, logical last-use clock, prediction probability, and
prefetch confidence. The eviction score remains:

```text
0.4 * normalized_frequency
+ 0.3 * normalized_recency
+ 0.2 * prediction_probability
+ 0.1 * prefetch_confidence
```

LRU remains available by setting `PackedRuntimeConfiguration::adaptiveResidency`
to false. Prediction can likewise be disabled independently for controlled
before/after measurements.

`RuntimeMetricsSnapshot` is a read-only internal API. It combines resident
expert counts, RAM/VRAM accounting, expert frequency, predictor quality, cache
hit rate, synchronization count, transfer latency, and kernel latency, and can
be serialized as `hypermoe.runtime-metrics.v1`. It neither starts a server nor
owns a UI.

CUDA execution now distinguishes a full backend synchronization from an
execution-stream boundary. Expert sub-operations enqueue work without inserting
their own full three-stream barrier. A transformer block completes its compute
dependency chain once through `synchronizeExecution`; transfer and prefetch
streams are not stopped by that boundary. Explicit host materialization and
full backend synchronization retain their existing correctness semantics.

`hypermoe_runtime_optimization_benchmark` runs the same packed artifact and
token sequence in baseline (LRU, no prediction) and adaptive modes. It reports
measured wall latency, expert reads/bytes, cache hit rate, prefetch outcomes,
and synchronization count. Results are hardware- and artifact-specific and
must not be generalized to an RTX 4070 unless run on that machine.
