#pragma once

#include "tensor/DType.hpp"
#include "tensor/Shape.hpp"
#include "tensor/quantization/Quantization.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace hypermoe::models {

struct TensorMetadata {
    TensorMetadata() = default;
    TensorMetadata(
        std::string tensorName,
        tensor::Shape tensorShape,
        tensor::DType tensorDType,
        std::optional<tensor::quantization::QuantizedDType> quantizedType,
        std::uint64_t tensorOffset,
        std::uint64_t tensorSize,
        std::uint32_t tensorLayerId,
        std::optional<std::uint32_t> tensorExpertId,
        std::optional<tensor::quantization::QuantizationParameters> parameters = {})
        : name(std::move(tensorName)),
          shape(std::move(tensorShape)),
          dtype(tensorDType),
          quantizedDType(quantizedType),
          offset(tensorOffset),
          size(tensorSize),
          layerId(tensorLayerId),
          expertId(tensorExpertId),
          quantizationParameters(parameters) {}

    std::string name;
    tensor::Shape shape;
    tensor::DType dtype{tensor::DType::FP32};
    std::optional<tensor::quantization::QuantizedDType> quantizedDType;
    std::uint64_t offset{};
    std::uint64_t size{};
    std::uint32_t layerId{};
    std::optional<std::uint32_t> expertId;
    std::optional<tensor::quantization::QuantizationParameters>
        quantizationParameters;

    [[nodiscard]] bool isQuantized() const noexcept {
        return quantizedDType.has_value() || quantizationParameters.has_value();
    }
};

} // namespace hypermoe::models
