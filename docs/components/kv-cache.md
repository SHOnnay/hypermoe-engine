# KV cache

`KVCache` owns correctness-first CPU key/value storage for every transformer
layer. Construction fixes layer count, maximum sequence length, key/value head
count, and head dimension. Appends carry their first absolute sequence position
and must be contiguous; gaps, overwrites, incompatible tensors, and capacity
overflow fail before storage changes.

Snapshots copy positions, keys, and values under a lock. This is intentionally
simple and safe for reference validation, including incremental attention. The
cache reports actual occupied bytes and supports per-layer clearing or full
reset.

No paged layout, eviction, beam sharing, device residency, or quantization is
claimed. A future paged cache can implement the same layer/position semantics
while delegating pages to the existing RAM/VRAM hierarchy.
