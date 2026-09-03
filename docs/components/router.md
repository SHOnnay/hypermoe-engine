# Router

`RouterConfig` specifies expert count, arbitrary top-k, score normalization, and
whether selected scores are renormalized. `RouterDecision` contains layer ID,
selected local expert IDs, and corresponding scores.

`RouterBackend` owns the scoring and selection implementation. The CPU reference
expects contiguous FP32 hidden states and a `[hidden_size, expert_count]`
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

The single-token API remains available. The batch API returns per-token decisions
plus expert groups containing token indices and routing weights. CUDA routing,
capacity limits, grouped expert execution, shared experts, and overflow/drop
policies are future backend capabilities.

`ExpertHistory` receives completed decisions. `TransitionDatabase` and
`ExpertPredictor` turn per-stream transitions, frequency, recency, and
co-occurrence into scheduler prefetch requests without coupling prediction to the
router backend.
