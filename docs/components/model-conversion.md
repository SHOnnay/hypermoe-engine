# Model conversion

`hypermoe_model_convert` is the offline boundary between a supported distribution
artifact and the runtime store. It asks `QwenImporter` to inspect `config.json`
and SafeTensors headers, then consumes only the returned v2 manifest. Qwen tensor
names are never interpreted by the packer or runtime.

For each `(layer_id, expert_id)`, the packer range-reads gate, up, and down
projections, validates their declared shape and dtype, and converts matrices from
`OUTPUT_INPUT` to the runtime's `INPUT_OUTPUT` convention. The three converted
projections are written contiguously in one 4 KiB-aligned expert payload. Router
matrices are converted and appended after expert payloads.

The output directory contains:

```text
hypermoe_model/
  manifest.json
  experts.bin
  experts.index
  conversion_report.json
```

Index v2 keeps the Phase 2 32-byte expert records intact and appends fixed-width
projection records. Each projection record contains layer, expert, role, byte
range, dtype, rank, shape, and CRC32. Both whole-expert and projection checksums
are validated. Stores with no projection extension are still written/read as the
legacy v1 format.

The converter refuses to overwrite an output directory. If conversion fails, it
removes only the new incomplete directory. It does not mutate the source model.

Usage:

```sh
./build/hypermoe_model_convert /path/to/qwen-moe /path/to/hypermoe_model
```

The current converter handles uncompressed FP32, FP16, BF16, and INT8 matrix
bytes described by the manifest. INT8 metadata can be packed, but quantized
expert execution still requires scale/group metadata and is intentionally
rejected by the executor.

The conversion report records layer, expert, projection, shard, source tensor,
parameter, byte, and dtype counts. Conversion completes only after reopening the
output store and validating every whole-expert and projection checksum.
