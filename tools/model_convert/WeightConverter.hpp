#pragma once

#include "models/ModelManifest.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace hypermoe::conversion {

struct ConvertedWeight {
    std::vector<std::byte> bytes;
    tensor::Shape shape;
    tensor::DType dtype{tensor::DType::FP32};
    models::TensorLayout layout{models::TensorLayout::InputOutput};
};

class WeightConverter {
public:
    [[nodiscard]] static ConvertedWeight convert(
        std::span<const std::byte> source,
        const tensor::Shape& shape,
        tensor::DType dtype,
        models::TensorLayout sourceLayout,
        models::TensorLayout targetLayout = models::TensorLayout::InputOutput);
};

} // namespace hypermoe::conversion
