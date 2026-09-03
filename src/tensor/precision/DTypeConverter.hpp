#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::tensor::precision {

class DTypeConverter {
public:
    [[nodiscard]] static std::vector<float>
    toFp32(std::span<const std::byte> source, DType sourceType);
    [[nodiscard]] static Tensor
    toFp32Tensor(TensorView source, TensorBackend& destinationBackend);
};

} // namespace hypermoe::tensor::precision
