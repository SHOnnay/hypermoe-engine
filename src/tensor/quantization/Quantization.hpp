#pragma once

#include "tensor/Shape.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hypermoe::tensor::quantization {

enum class QuantizedDType : std::uint32_t {
    INT8 = 1,
    Q4 = 2,
};

struct QuantizationParameters {
    float scale{1.0F};
    std::int32_t zeroPoint{};
};

[[nodiscard]] constexpr std::string_view toString(QuantizedDType dtype) noexcept {
    switch (dtype) {
    case QuantizedDType::INT8: return "INT8";
    case QuantizedDType::Q4: return "Q4";
    }
    return "UNKNOWN";
}

[[nodiscard]] std::size_t storageSizeBytes(const Shape& shape,
                                           QuantizedDType dtype);
void validateParameters(QuantizedDType dtype,
                        const QuantizationParameters& parameters);

} // namespace hypermoe::tensor::quantization
