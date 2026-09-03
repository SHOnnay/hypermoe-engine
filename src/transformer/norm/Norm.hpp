#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <string_view>

namespace hypermoe::transformer::norm {

class Norm {
public:
    virtual ~Norm() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual tensor::Device device() const noexcept = 0;
    [[nodiscard]] virtual tensor::Tensor execute(
        tensor::TensorView input,
        tensor::TensorView weight) = 0;
};

} // namespace hypermoe::transformer::norm
