# Transformer runtime foundation

`TransformerLayer` is the generic layer execution boundary. `MoELayer` supplies
the first implementation and connects a validated hidden state to `MoERuntime`,
which routes, schedules, adopts, executes, and combines selected experts.

The current flow is:

```text
hidden state → validated identity attention placeholder
             → router → selected experts → weighted MoE output
             → residual add → layer output
```

The attention placeholder performs a real checked tensor copy rather than
pretending attention was computed. This makes tensor ownership, device, dtype,
shape, and residual behavior testable while leaving attention architecture and
KV-cache policy unspecified.

Phase 10 validates one BF16-backed Qwen-style MoE layer using FP32 CPU execution
and an independent scalar oracle. There is no normalization, attention kernel,
tokenizer, generation loop, or server yet.
