# Precision runtime

Storage dtype and execution dtype are separate contracts. Import and packing
preserve FP32, FP16, or BF16 bytes and metadata. The CPU reference runtime uses
`DTypeConverter` to materialize selected FP16/BF16 expert projections as FP32
immediately before execution; it never converts the complete checkpoint.

FP16 conversion handles normals, subnormals, signed zero, infinities, and NaNs.
BF16 conversion reconstructs the IEEE FP32 bit pattern. Source ranges must be
non-empty and exactly aligned to their dtype width. INT8 conversion is rejected
without quantization metadata.

`QuantizationPolicy` records INT8, Q4, or Q8 type, scale, zero point, and group
size. It validates zero-point range and requires the group size to divide a
contiguous tensor. This is a metadata contract only: optimized quantized GEMM and
model-specific scale tensors remain future work.

The current CUDA path still expects FP32 execution tensors. A future CUDA
precision stage can implement the same storage-to-execution boundary using
device conversion or native cuBLAS precision modes.
