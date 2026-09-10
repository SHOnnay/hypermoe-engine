# KV cache runtime

`KVCache` stores CPU FP32 keys and values by layer, key/value head, head feature,
and contiguous sequence position. Vectors grow as tokens arrive, while append
operations reserve all affected arrays before changing their logical sizes. A
failed allocation therefore cannot leave keys, values, and positions at
different lengths.

`KVCacheManager` owns the session budget. It calculates the maximum logical
reservation for one cache as:

```text
layers * maximum_sequence *
    (sizeof(position) + 2 * key_value_heads * head_dimension * sizeof(float))
```

A session is admitted only when that full reservation fits under the configured
memory limit. This conservative admission rule prevents dynamic cache growth
from creating unbounded aggregate memory demand. Statistics report active
sessions, committed logical bytes, reserved bytes, peak committed bytes, and the
configured limit.

The current accounting excludes allocator capacity overhead and stores snapshots
by value for the CPU attention reference path. Future paged GPU caches should
preserve the manager's admission and ownership contracts while replacing the
physical layout and snapshot mechanism.
