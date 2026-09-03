#include "tensor/quantization/Quantization.hpp"

#include <cmath>
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

} // namespace hypermoe::tensor::quantization
