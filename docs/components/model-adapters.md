# Model adapters

## Contract

`ModelAdapter` exposes metadata loading, tensor-index loading, architecture labels,
capabilities, layer/expert counts, router configuration, and generic expert weight
mappings. Runtime consumers depend on capabilities and neutral structures rather
than architecture labels.

The neutral graph contains:

- `ModelConfig`: counts, widths, name, and capabilities.
- `TensorMetadata`: name, checked shape, dense dtype, optional packed encoding,
  64-bit offset/size, layer ID, and optional expert ID.
- `LayerMetadata`: router tensor and the experts discovered for one layer.
- `ExpertMetadata`: the source tensors belonging to one local expert.

## HyperMoE manifest v1

The first adapter consumes a JSON inspection manifest with this shape:

```json
{
  "schema": "hypermoe.model-manifest.v1",
  "architecture": "QWEN_MOE",
  "model_name": "inspection label",
  "layer_count": 32,
  "expert_count": 128,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "router": {
    "expert_count": 128,
    "top_k": 8,
    "normalization": "SOFTMAX",
    "renormalize_selected": true
  },
  "tensors": []
}
```

Each tensor entry contains `name`, `shape`, `dtype`, `offset`, `size`, `layer_id`,
and nullable `expert_id`. Offsets are absolute within the associated weight store.
Dense dtypes are `FP32`, `FP16`, and `INT8`; `Q4` and optional `quantization`
metadata describe packed storage but are not executable yet.

This manifest is not a native model format. A native GGUF or SafeTensors reader
must validate actual source metadata, then populate these structures.

## Current adapter

`QwenMoEAdapter` recognizes only Qwen-style names inside its own translation unit:

```text
model.layers.<layer>.mlp.experts.<expert>.gate_proj.weight
model.layers.<layer>.mlp.experts.<expert>.up_proj.weight
model.layers.<layer>.mlp.experts.<expert>.down_proj.weight
model.layers.<layer>.mlp.gate.weight
```

It cross-checks IDs, counts, shapes, dtypes, complete projection groups, router
presence, and non-overlapping ranges. Other families should implement independent
adapters rather than adding name branches to the core.
