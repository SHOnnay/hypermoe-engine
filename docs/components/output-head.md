# Output normalization and LM head

`FinalNorm` is an independent wrapper around the reference RMSNorm operation.
Its hidden dimension and epsilon come from the manifest architecture, while its
scale vector is resolved through the `final_norm` binding.

`LMHead` maps a rank-2 hidden-state tensor to rank-2 vocabulary logits. A
separate packed head uses `INPUT_OUTPUT` shape `[hidden, vocabulary]`. A tied
head uses the shared embedding table in `OUTPUT_INPUT` shape
`[vocabulary, hidden]`; the scalar reference projection accounts for that
orientation without allocating or transposing another table.

Both paths currently require contiguous CPU FP32 tensors. There is no output
bias, softmax, sampling, vocabulary filtering, or CUDA implementation. Those
operations are intentionally separate from the raw-logit contract.
