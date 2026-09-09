#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hypermoe::tensor {

enum class DType : std::uint32_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
};

[[nodiscard]] constexpr bool isValid(DType dtype) noexcept {
    switch (dtype) {
    case DType::FP32:
    case DType::FP16:
    case DType::BF16:
    case DType::INT8: return true;
    }
    return false;
}

[[nodiscard]] constexpr std::size_t alignmentOf(DType dtype) noexcept {
    switch (dtype) {
    case DType::FP32: return alignof(float);
    case DType::FP16:
    case DType::BF16: return alignof(std::uint16_t);
    case DType::INT8: return alignof(std::int8_t);
    }
    return 0;
}

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
