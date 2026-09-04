# Model runtime

`models::runtime::ModelArchitecture` is the validated execution description
derived from a `ModelManifest`. It records layer and hidden dimensions,
query/key-value head counts, head width, expert count, top-k, RoPE theta, and the
input/post-attention normalization configurations. Runtime construction rejects
metadata that disagrees with the manifest's model or router dimensions.

`ManifestLayerMapping` binds logical roles—Q/K/V/O projections, input norm,
post-attention norm, and router—to neutral tensor names and declared layouts.
Only the Qwen importer recognizes upstream Qwen names. Offline packing converts
matrix tensors to the row-major `INPUT_OUTPUT` execution layout and rewrites the
bindings; transformer runtime code never parses a model-family path.

`RuntimeTensorMap` owns the shared tensors referenced by those bindings.
`TransformerModelRuntime` constructs checked block weights, applies causal RoPE
attention, both residual branches, normalization, grouped MoE execution, and
passes each output to the next layer. Per-layer results retain routing,
execution timings, and output tensors for correctness inspection.

The runtime currently accepts already materialized FP32 shared tensors. Loading,
caching, and precision conversion for non-expert tensors should be unified with
the hierarchical memory system in a later phase.
