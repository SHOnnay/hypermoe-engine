# Adaptive expert predictor

Phase 21 keeps expert prediction statistical and deterministic. It does not
inspect token text or introduce a learned routing model. The predictor consumes
the current layer and selected expert IDs, then combines four bounded signals
for the next layer:

- decayed source-to-target transition probability;
- popularity inside a fixed recent-observation window;
- decayed, layer-local lifetime popularity; and
- same-ID locality across adjacent layers; and
- within-layer expert co-occurrence.

`TransitionDatabase` stores exact counters for diagnostics and lazy-decayed
counters for decisions. Lazy decay records the observation number at each
update, so old evidence loses influence without an O(number-of-experts) sweep.
The recent window is bounded and the database remains partitioned by inference
stream when transitions are recorded.

`ExpertPrediction` separates probability from confidence. Probability ranks
candidates relative to the other experts in the expected layer. Confidence
describes the amount and agreement of available evidence. Sorting is by
probability, confidence, then expert ID, making equal-score behavior stable
across supported standard libraries.

The prefetch threshold is independent of the prediction threshold. Predictions
below the transfer threshold can still update residency scores, while avoiding
an NVMe or PCIe movement whose evidence is weak. The predictor reports top-1
accuracy, top-k coverage, and per-expert prefetch usefulness. These values are
observational metrics, not claims about model quality.

Run `hypermoe_prediction_benchmark [report.json]` for the deterministic
synthetic comparison. Its no-prediction baseline intentionally issues no
prefetch requests; the adaptive result reports measured prediction latency and
quality on the same routing sequence.
