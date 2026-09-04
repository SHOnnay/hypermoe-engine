# HyperMoE model manifest

`hypermoe.model-manifest.v2` is the validated handoff between artifact import and
runtime preparation. It contains:

- diagnostic model and source architecture identifiers;
- layer, expert, hidden, and expert-intermediate sizes;
- explicit runtime capability flags, including shared and quantized experts;
- top-k normalization and selected-score renormalization;
- source-file-relative tensor locations, shapes, dtypes, offsets, and sizes;
- layer-addressed router tensor names and matrix orientation;
- one gate/up/down projection mapping for each discovered layer/expert identity.

Physical tensors and logical projections are separate. A Qwen3 fused gate/up
tensor appears once in the physical index, while each expert mapping references
its checked byte slices. `OUTPUT_INPUT` records the source linear-weight layout;
the importer does not pretend those bytes already match HyperMoE's current GEMM
orientation.

Loading validates schema version, unique names and expert identities, model
bounds, relative paths, overflow, projection containment, shape-derived byte
sizes, and all router/mapping references. The JSON round trip preserves 64-bit
file offsets.

The earlier v1 adapter-validation manifest remains supported by
`QwenMoEAdapter`; v2 is the artifact import/runtime handoff and is intentionally
more explicit.

Phase 13 extends v2 compatibly with optional `runtime_architecture` metadata and
a `layers` array. Each complete layer binds Q/K/V/O projections, input and
post-attention normalization, and its router tensor by logical role. Matrix
bindings carry an explicit layout. Existing expert-only v2 manifests remain
valid; `TransformerModelRuntime` requires the complete extension.
