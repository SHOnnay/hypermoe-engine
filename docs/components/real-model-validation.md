# Real-model validation

Phase 19 provides a model-neutral validation framework for Qwen-compatible MoE
artifacts. It does not bundle a checkpoint and does not infer missing metadata.

## Preparation pipeline

`prepareQwenArtifact` runs the Qwen importer, validates discovered tensor ranges
and shapes against the physical checkpoint, packs expert projections into the
portable HyperMoE store, reloads the generated manifest, and requires complete
architecture, model-I/O, layer, and expert mappings. The runtime consumes only
that generated manifest.

The real-model benchmark accepts an artifact path and a separate packed-output
directory. It records measured import/validation/packing time and packed bytes.
Fields that require an actual forward/generation run—TTFT, generation speed,
expert-transfer statistics, and cache hit rate—are emitted as `null` until a
caller supplies a runnable checkpoint fixture and execution trace.

## CPU/CUDA oracle

`compareRealModelTraces` validates shape and value agreement for final logits,
every transformer-layer output, and every captured expert output. FP32 comparison
uses the existing correctness-oracle tolerance. Structural mismatches fail before
numeric comparison, which makes missing trace stages explicit.

Target RTX validation must use the same artifact, prompt, and token IDs on CPU
and CUDA. Reports should include the checkpoint identity, build configuration,
driver/toolkit versions, and measured values; Mac validation covers CPU fallback
only.
