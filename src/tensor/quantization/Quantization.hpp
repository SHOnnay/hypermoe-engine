#pragma once

#include "tensor/Shape.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace hypermoe::tensor::quantization {

enum class QuantizedDType : std::uint32_t {
    INT8 = 1,
    Q4 = 2,
    Q8 = 3,
};

struct QuantizationParameters {
    float scale{1.0F};
    std::int32_t zeroPoint{};

    friend bool operator==(const QuantizationParameters&,
                           const QuantizationParameters&) = default;
};

struct Int8QuantizationResult {
    std::vector<std::byte> bytes;
    QuantizationParameters parameters;
    float maximumAbsoluteError{};
};

[[nodiscard]] constexpr std::string_view toString(QuantizedDType dtype) noexcept {
    switch (dtype) {
    case QuantizedDType::INT8: return "INT8";
    case QuantizedDType::Q4: return "Q4";
    case QuantizedDType::Q8: return "Q8";
    }
    return "UNKNOWN";
}

[[nodiscard]] std::size_t storageSizeBytes(const Shape& shape,
                                           QuantizedDType dtype);
void validateParameters(QuantizedDType dtype,
                        const QuantizationParameters& parameters);
[[nodiscard]] Int8QuantizationResult quantizeInt8(std::span<const float> values);
[[nodiscard]] std::vector<float> dequantizeInt8(
    std::span<const std::byte> values,
    const QuantizationParameters& parameters);

} // namespace hypermoe::tensor::quantization
