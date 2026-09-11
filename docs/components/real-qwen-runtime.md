# Real Qwen runtime

Phase 20 connects a Hugging Face-style Qwen MoE checkpoint to the existing
forward-to-logits graph. Qwen-specific names remain inside the importer and
checkpoint loader; execution consumes the packed HyperMoE manifest only.

## Accepted checkpoint inputs

The strict loader accepts a directory or SafeTensors file rooted beside:

- `config.json` identifying `Qwen2MoeForCausalLM`, `Qwen3MoeForCausalLM`, or the
  corresponding `qwen2_moe`/`qwen3_moe` model type;
- one SafeTensors file, multiple discovered shards, or
  `model.safetensors.index.json` with a complete safe relative `weight_map`;
- `tokenizer.json` and `tokenizer_config.json`; and
- optional `special_tokens_map.json`.

Every manifest range, dtype, shape, expert projection, router tensor, attention
projection, normalization tensor, embedding, and output mapping is checked
against the physical shard metadata. Tokenizer vocabulary and added-token IDs,
plus numeric BOS/EOS/PAD IDs from `config.json`, must fit `vocab_size`.
Optional Qwen3 per-head `q_norm` and `k_norm` weights are mapped as a required
pair and executed before RoPE on both CPU and CUDA paths.

## Conversion and execution

`hypermoe_real_checkpoint_convert` writes a new directory containing the packed
manifest, `experts.bin`, `experts.index`, conversion report, normalized tokenizer
metadata, and copied tokenizer assets. Existing output directories are never
overwritten. A failure after packing removes only the newly created output.

`PackedModelRuntime` memory-maps the packed data, loads only persistent attention,
router, normalization, embedding, and LM-head tensors, and converts FP16/BF16
storage to FP32 execution tensors. Experts remain cold until routing schedules
them through the existing NVMe/RAM/pinned/device hierarchy. CPU and CUDA select
different backends without changing the model graph.

`forward()` accepts one sequence of token IDs and returns embeddings, attention
and expert intermediates, layer outputs, normalized hidden states, and logits.
An optional bounded KV cache supports position-by-position measurement.

## Current compatibility

Qwen3 routed MoE layouts and Qwen2 individual routed-expert layouts are
recognized. Qwen2 checkpoints declaring shared experts are validated and can be
converted, but execution fails explicitly because the neutral shared-expert
residual is not implemented. Dense/non-MoE layers, projection bias, quantized
checkpoint formats, native tokenizer encoding, batching, and sampling are not
part of this phase.
