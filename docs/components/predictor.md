# Expert predictor

`TransitionDatabase` records routing observations without a machine-learning
model. Its bounded and thread-safe state includes expert frequency, cross-layer
transitions, top-k co-occurrence, recent selections, and previous selections per
inference stream.
Call `endStream` when a sequence finishes so per-stream predecessor state does not
grow with completed requests.

`ExpertPredictor` ranks experts observed in the next layer using:

```text
0.70 × transition probability
+ 0.20 × layer frequency
+ 0.10 × same-ID recency
+ 0.05 × normalized co-occurrence boost
```

Scores are clamped to `[0, 1]`, confidence-filtered, sorted deterministically,
and limited by configuration. The coefficients are an initial policy for
measurement, not a learned property of Qwen or another model.

`observeAndPrefetch` connects a router decision to `ExpertHistory`, the transition
database, and scheduler prefetch requests. Unknown experts in a partial runtime
graph are skipped. Active demand can upgrade pending speculative work, and
completed prefetch storage remains scheduler-owned so a later demand request can
reuse the actual buffer.

Future predictors can implement the existing `Prefetcher` interface using router
logits, token context, or a learned model without changing transfer priority or
residency state contracts.
