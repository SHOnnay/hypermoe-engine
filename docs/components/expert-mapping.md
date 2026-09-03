# Expert mapping

`ExpertWeightMap` is the boundary between source tensor names and generic expert
execution. Adapters add each tensor under one of three roles:

- `GATE`
- `UP`
- `DOWN`

Mappings are keyed by `(layer_id, expert_id)`. Duplicate roles, identity
mismatches, incomplete experts, invalid byte sizes, offsets before the payload,
and slices outside the resident buffer are rejected.

At execution time, the scheduler's index record supplies the expert payload's
absolute file offset. `ExpertWeightMap::createViews` subtracts that base and
creates read-only `TensorView` slices directly over the resident `DeviceBuffer`.
No model-specific string reaches `ExpertMlpExecutor`, and no projection weight is
copied merely to change its metadata representation.

Packed INT8/Q4 tensors can be represented in metadata, but view creation currently
rejects them because the FP32 executor has no validated dequantization layout.
Future adapters must provide block/group scale layout and packing semantics before
quantized execution is enabled.
