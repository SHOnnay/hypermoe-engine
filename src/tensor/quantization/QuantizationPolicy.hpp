#pragma once

#include "tensor/Shape.hpp"
#include "tensor/quantization/Quantization.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace hypermoe::tensor::quantization {

struct QuantizationPolicy {
    QuantizedDType tensorType{QuantizedDType::INT8};
    float scale{1.0F};
    std::int32_t zeroPoint{};
    std::size_t groupSize{1};

    void validate(const Shape& shape) const;
    [[nodiscard]] std::string toJson() const;
};

} // namespace hypermoe::tensor::quantization
