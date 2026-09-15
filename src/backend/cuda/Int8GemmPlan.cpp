#include "backend/cuda/Int8GemmPlan.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hypermoe::backend::cuda {
namespace {
std::size_t product(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error("INT8 GEMM launch size overflow");
    }
    return left * right;
}
std::size_t divideUp(std::size_t value, std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}
} // namespace

std::string_view toString(Int8GemmMode mode) noexcept {
    switch (mode) {
    case Int8GemmMode::Auto: return "auto";
    case Int8GemmMode::Reference: return "reference";
    case Int8GemmMode::Cooperative: return "cooperative";
    }
    return "invalid";
}

Int8GemmPlan planInt8Gemm(std::size_t rows, std::size_t inner,
                          std::size_t columns, Int8GemmMode mode) {
    if (rows == 0 || inner == 0 || columns == 0 || toString(mode) == "invalid") {
        throw std::invalid_argument("INT8 GEMM dimensions/mode are invalid");
    }
    const auto elements = product(rows, columns);
    (void)product(rows, inner);
    (void)product(inner, columns);
    Int8GemmPlan result;
    result.implementation = mode == Int8GemmMode::Auto
        ? (inner >= 256 && columns >= 64 ? Int8GemmMode::Cooperative : Int8GemmMode::Reference)
        : mode;
    if (result.implementation == Int8GemmMode::Reference) {
        result.blocks = divideUp(elements, 256);
    } else {
        result.partitions = std::min<std::size_t>(8, divideUp(inner, 512));
        result.blocks = product(product(rows, result.partitions), divideUp(columns, 32));
        if (result.partitions > 1) result.scratchElements = product(elements, result.partitions);
    }
    if (result.blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        result.scratchElements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("INT8 GEMM exceeds CUDA launch/address limits");
    }
    return result;
}
} // namespace hypermoe::backend::cuda
