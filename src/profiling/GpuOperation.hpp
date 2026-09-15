#pragma once

namespace hypermoe::profiling {
enum class GpuOperation {
    Fp32Gemm, Int8Gemm, ExpertFp32Gemm, ExpertInt8Gemm,
    Activation, Elementwise, RMSNorm, RoPE, Router, AttentionCore,
    Gather, Scatter, Zero, AttentionRegion, ExpertRegion
};
} // namespace hypermoe::profiling
