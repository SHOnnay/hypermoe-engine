#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <cstddef>
#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}
namespace hypermoe::transformer::norm {
class RMSNorm;
}

namespace hypermoe::transformer::output {

class FinalNorm {
public:
    FinalNorm(std::shared_ptr<tensor::TensorBackend> backend,
              std::size_t hiddenDimension,
              float epsilon);

    [[nodiscard]] tensor::Tensor execute(tensor::TensorView hiddenStates,
                                         tensor::TensorView weights) const;
    [[nodiscard]] std::size_t hiddenDimension() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;

private:
    std::shared_ptr<norm::RMSNorm> implementation_;
};

} // namespace hypermoe::transformer::output
