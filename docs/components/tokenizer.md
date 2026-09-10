# Tokenizer boundary

`Tokenizer` is an architecture-independent interface with `encode`, `decode`,
and `vocabularySize`. The generation runtime depends only on this interface and
checks that its vocabulary agrees with the model architecture.

`QwenTokenizerAdapter` is the first model-family integration seam. It accepts
validated encoder and decoder callbacks supplied by a tokenizer implementation.
Every returned or decoded token ID is checked against the declared vocabulary.

This phase deliberately does not claim to parse Qwen `tokenizer.json`, execute
byte-pair encoding, apply a chat template, or interpret special-token files.
Those behaviors depend on real artifact metadata and belong in a future
tokenizer provider behind the adapter. Tests use a deterministic fixture
vocabulary so generation behavior remains reproducible on libc++ and MSVC.
