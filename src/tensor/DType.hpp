#pragma once

#include <cstddef>
#include <string_view>

namespace hypermoe::tensor {

enum class DType {
    FP32,
    FP16,
    BF16,
    INT8,
};

[[nodiscard]] constexpr std::size_t sizeOf(DType dtype) noexcept {
    switch (dtype) {
    case DType::FP32: return 4;
    case DType::FP16: return 2;
    case DType::BF16: return 2;
    case DType::INT8: return 1;
    }
    return 0;
}

[[nodiscard]] constexpr std::string_view toString(DType dtype) noexcept {
    switch (dtype) {
    case DType::FP32: return "FP32";
    case DType::FP16: return "FP16";
    case DType::BF16: return "BF16";
    case DType::INT8: return "INT8";
    }
    return "UNKNOWN";
}

} // namespace hypermoe::tensor
