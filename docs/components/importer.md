# Model importer

`ModelImporter` separates source-artifact discovery from runtime consumption. An
importer inspects metadata and tensor indexes, validates architecture-specific
meaning, and emits a `ModelManifest`. It does not allocate VRAM, schedule experts,
or execute tensors.

The architecture-neutral SafeTensors reader:

- reads only the eight-byte header length and bounded JSON header;
- supports single-file and sharded artifact directories;
- records shard-relative paths and absolute in-file data offsets;
- checks shape × dtype against every declared range;
- rejects duplicate tensors, unsupported dtypes, truncation, overflow, and paths
  outside the artifact root.

`QwenImporter` reads `config.json`, requires explicit routing fields, and accepts
recognized Qwen2/Qwen3 MoE architecture metadata. Qwen naming and fused-storage
rules exist only in that importer. It understands separate per-expert projections
and the current fused expert collection layout. Dense model tensors are omitted
from the expert manifest because Phase 8 does not implement transformer layers.

Import currently performs metadata indexing only. It does not download models,
rewrite SafeTensors, generate checksums for source payloads, or pack `experts.bin`.
`hypermoe_model_import` exposes the current importer as a command-line tool;
`hypermoe_model_inspect` can inspect either the source artifact or emitted v2
manifest without opening tensor payloads.
