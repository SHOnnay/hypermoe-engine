#include "tensor/quantization/Quantization.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hypermoe::tensor::quantization {

std::size_t storageSizeBytes(const Shape& shape, QuantizedDType dtype) {
    if (!shape.isContiguous()) {
        throw std::invalid_argument("packed quantized tensors require contiguous shapes");
    }
    const auto elements = shape.elementCount();
    switch (dtype) {
    case QuantizedDType::INT8:
    case QuantizedDType::Q8: return elements;
    case QuantizedDType::Q4: return elements / 2 + elements % 2;
    }
    throw std::invalid_argument("unsupported quantized dtype");
}

void validateParameters(QuantizedDType dtype,
                        const QuantizationParameters& parameters) {
    if (!std::isfinite(parameters.scale) || parameters.scale <= 0.0F) {
        throw std::invalid_argument("quantization scale must be finite and positive");
    }
    switch (dtype) {
    case QuantizedDType::INT8:
    case QuantizedDType::Q8:
        if (parameters.zeroPoint < -128 || parameters.zeroPoint > 127) {
            throw std::invalid_argument("8-bit zero point is outside [-128, 127]");
        }
        return;
    case QuantizedDType::Q4:
        if (parameters.zeroPoint < -8 || parameters.zeroPoint > 7) {
            throw std::invalid_argument("Q4 zero point is outside [-8, 7]");
        }
        return;
    }
    throw std::invalid_argument("unsupported quantized dtype");
}

Int8QuantizationResult quantizeInt8(std::span<const float> values) {
    if (values.empty()) {
        throw std::invalid_argument("INT8 quantization source cannot be empty");
    }
    float maximumMagnitude{};
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "INT8 quantization source must contain finite values");
        }
        maximumMagnitude = std::max(maximumMagnitude, std::fabs(value));
    }

    Int8QuantizationResult result;
    result.parameters.scale = maximumMagnitude == 0.0F
        ? 1.0F
        : maximumMagnitude / 127.0F;
    result.parameters.zeroPoint = 0;
    validateParameters(QuantizedDType::INT8, result.parameters);
    result.bytes.resize(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto rounded = std::round(values[index] / result.parameters.scale);
        const auto clamped = std::clamp(rounded, -127.0F, 127.0F);
        const auto quantized = static_cast<std::int8_t>(clamped);
        result.bytes[index] = std::bit_cast<std::byte>(quantized);
        const auto reconstructed =
            static_cast<float>(static_cast<std::int32_t>(quantized) -
                               result.parameters.zeroPoint) *
            result.parameters.scale;
        result.maximumAbsoluteError = std::max(
            result.maximumAbsoluteError,
            std::fabs(values[index] - reconstructed));
    }
    return result;
}

std::vector<float> dequantizeInt8(
    std::span<const std::byte> values,
    const QuantizationParameters& parameters) {
    if (values.empty()) {
        throw std::invalid_argument("INT8 dequantization source cannot be empty");
    }
    validateParameters(QuantizedDType::INT8, parameters);
    std::vector<float> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto quantized = std::bit_cast<std::int8_t>(values[index]);
        result[index] =
            static_cast<float>(static_cast<std::int32_t>(quantized) -
                               parameters.zeroPoint) *
            parameters.scale;
    }
    return result;
}

} // namespace hypermoe::tensor::quantization
