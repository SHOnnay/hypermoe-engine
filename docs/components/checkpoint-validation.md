# Checkpoint validation

`SafeTensorShardManager` presents one logical tensor namespace over a single
SafeTensors file or a Hugging Face sharded checkpoint. When
`model.safetensors.index.json` exists, only declared shards are opened and every
`weight_map` entry must resolve to the shard that actually contains the tensor.
Without an index, all `.safetensors` files in the artifact directory are
discovered deterministically.

Each shard header is bounded and parsed independently. Shape, dtype, byte count,
non-overlapping ranges, file bounds, duplicate global names, relative paths, and
index mappings are validated without loading tensor payloads. The manager then
supports checked tensor-relative range reads from the correct shard.

`CheckpointValidator` compares an imported HyperMoE manifest back to this global
index. File, offset, size, shape, and dtype must match exactly, while normal
manifest validation guarantees complete router and gate/up/down expert mappings.
Its report includes shard/tensor/expert counts, referenced bytes, and dtype
statistics.

Current native artifact support is Qwen2/Qwen3 MoE SafeTensors. GGUF, remote Hub
fetching, and other model families remain unsupported.
