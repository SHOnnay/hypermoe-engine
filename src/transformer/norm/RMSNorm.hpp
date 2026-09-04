#pragma once

#include "transformer/norm/Norm.hpp"

#include <cstddef>
#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::transformer::norm {

class RMSNorm final : public Norm {
public:
    RMSNorm(std::shared_ptr<tensor::TensorBackend> backend,
            std::size_t hiddenDimension,
            float epsilon = 1.0e-6F);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] tensor::Device device() const noexcept override;
    [[nodiscard]] tensor::Tensor execute(
        tensor::TensorView input,
        tensor::TensorView weight) override;
    [[nodiscard]] std::size_t hiddenDimension() const noexcept override;
    [[nodiscard]] float epsilon() const noexcept override;

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
    std::size_t hiddenDimension_{};
    float epsilon_{};
};

} // namespace hypermoe::transformer::norm
