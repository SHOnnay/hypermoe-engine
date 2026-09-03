#pragma once

#include "tensor/DType.hpp"
#include "tensor/Shape.hpp"
#include "tensor/quantization/Quantization.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace hypermoe::models {

struct TensorMetadata {
    std::string name;
    tensor::Shape shape;
    tensor::DType dtype{tensor::DType::FP32};
    std::optional<tensor::quantization::QuantizedDType> quantizedDType;
    std::uint64_t offset{};
    std::uint64_t size{};
    std::uint32_t layerId{};
    std::optional<std::uint32_t> expertId;

    [[nodiscard]] bool isQuantized() const noexcept {
        return quantizedDType.has_value();
    }
};

} // namespace hypermoe::models
