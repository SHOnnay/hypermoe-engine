# Model forward runtime

`models::runtime::ModelRuntime` is the complete Phase 14 execution boundary. It
accepts numeric token IDs and an `InferenceContext`, then executes:

```text
Embedding → TransformerModelRuntime → FinalNorm → LMHead → logits
```

Construction resolves all model-I/O tensor names from `ManifestModelIO` against
the existing owning `RuntimeTensorMap`. The components receive lifetime-checked
`TensorView` objects; they do not allocate another copy of model weights. Output
tensors use ordinary RAII ownership and remain valid in `ModelForwardResult`.

The result exposes embedding, transformer, final-normalization, LM-head, and
total wall-clock durations. It also retains intermediate tensors and per-layer
results for correctness diagnosis. The runtime validates non-empty token input,
batch size, hidden dimension, vocabulary bounds, tensor dtype/device, and exact
manifest shapes before arithmetic.

Phase 14 is a CPU FP32 reference path. Tokenization, iterative generation,
sampling, logits post-processing, CUDA embedding/output kernels, and memory-
releasing production execution modes are outside this component.
