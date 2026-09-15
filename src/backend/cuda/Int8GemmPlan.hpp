#pragma once

#include <cstddef>
#include <string_view>

namespace hypermoe::backend::cuda {

enum class Int8GemmMode { Auto, Reference, Cooperative };
[[nodiscard]] std::string_view toString(Int8GemmMode mode) noexcept;

struct Int8GemmPlan {
    Int8GemmMode implementation{Int8GemmMode::Reference};
    std::size_t partitions{1};
    std::size_t blocks{};
    std::size_t scratchElements{};
};

// CUDA-independent launch planning; no tensor layout/quantization changes.
[[nodiscard]] Int8GemmPlan planInt8Gemm(std::size_t rows, std::size_t inner,
                                       std::size_t columns, Int8GemmMode mode = Int8GemmMode::Auto);

} // namespace hypermoe::backend::cuda
