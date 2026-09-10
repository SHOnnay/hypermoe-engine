# Incremental generation

HyperMoE generation is an orchestration layer over the existing forward runtime.
It does not contain model-specific tensor names, tokenizer algorithms, or
sampling policy inside the transformer implementation.

```text
prompt text
    -> Tokenizer::encode
    -> Decoder::prefill (all prompt tokens)
    -> session KV cache + last-token logits
    -> Sampler
    -> Decoder::decode (one selected token)
    -> repeat until token limit or stop token
    -> Tokenizer::decode
```

`InferenceSession` owns one bounded KV cache allocation, `GenerationState`, and
`ForwardState`. The first output token is sampled from the last row of the
prefill logits. Each subsequent selected token is forwarded once at the next
sequence position. Session destruction releases its cache reservation.

`GenerationModel` isolates generation from a concrete model implementation.
`ModelRuntimeGenerationModel` connects it to the complete artifact-backed
`ModelRuntime`. Deterministic models can implement the same interface for tests
and benchmarks without constructing fake transformer internals.

`ForwardState` moves the latest hidden-state and logits owners into session
state. Consumers receive stable tensor ownership without copying their payloads.
It also records the final layer, sequence position, cache extent, and latest
routing decisions.

Generation currently supports one active sequence per session, CPU logits, and
greedy or seeded stochastic sampling. There is no batching across sessions,
beam search, speculative decoding, server, or streaming callback yet.
