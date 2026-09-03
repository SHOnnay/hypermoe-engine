# Batched MoE layer

The Phase 12 MoE path accepts a rank-2 token batch. The router produces one
normalized top-k decision per token plus deterministic groups keyed by expert.
Each group becomes an `ExpertBatch` containing the expert identity, unique token
indices, and matching routing weights.

```text
token batch → batch router → expert groups
                              │
                              ├─ request each unique expert once
                              ├─ gather assigned rows
                              ├─ execute one expert MLP per group
                              └─ weighted scatter-add into token outputs
```

All selected expert transfers are submitted before waiting, allowing scheduler
workers to overlap disk and device movement. A residency lease covers projection
views and execution. The CPU reference currently gathers and combines through
host tensors; the structure is intentionally compatible with future grouped
CUDA gather/scatter and GEMM without changing routing or scheduler contracts.

`CorrectnessOracle` independently recomputes token routing, every expert MLP,
selected-weight normalization, and the final weighted sum. It never calls the
runtime router or tensor backend when generating expected values.
