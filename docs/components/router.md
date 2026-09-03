# Router

`RouterConfig` specifies expert count, arbitrary top-k, score normalization, and
whether selected scores are renormalized. `RouterDecision` contains layer ID,
selected local expert IDs, and corresponding scores.

`RouterBackend` owns the scoring and selection implementation. The CPU reference
expects one contiguous FP32 hidden state and a `[hidden_size, expert_count]`
weight matrix. It computes logits in double-precision accumulation, rejects
non-finite values, optionally applies stable softmax, and resolves score ties by
ascending expert ID.

The generic layer pipeline is:

```text
hidden state + router weights
             │
             ▼
          Router
             │ RouterDecision
             ▼
          Scheduler
             │ ready DeviceBuffers
             ▼
 ExpertManager + ExpertWeightMap
             │ zero-copy views
             ▼
      ExpertMlpExecutor
             │ routing-weighted sum
             ▼
          output
```

The current router is CPU-only and routes one token per call. CUDA/batched routing,
capacity limits, token grouping, shared experts, and overflow/drop policies are
future backend capabilities.

`ExpertHistory` receives completed decisions and records frequency, transitions,
and previous selections. A future predictor can turn that state into scheduler
prefetch requests without coupling prediction to the router backend.
