#include "transformer/output/FinalNorm.hpp"

#include "transformer/norm/RMSNorm.hpp"

#include <utility>

namespace hypermoe::transformer::output {

FinalNorm::FinalNorm(std::shared_ptr<tensor::TensorBackend> backend,
                     std::size_t hiddenDimension,
                     float epsilon)
    : implementation_(std::make_shared<norm::RMSNorm>(
          std::move(backend), hiddenDimension, epsilon)) {}

tensor::Tensor FinalNorm::execute(tensor::TensorView hiddenStates,
                                  tensor::TensorView weights) const {
    return implementation_->execute(hiddenStates, weights);
}

std::size_t FinalNorm::hiddenDimension() const noexcept {
    return implementation_->hiddenDimension();
}

float FinalNorm::epsilon() const noexcept { return implementation_->epsilon(); }

} // namespace hypermoe::transformer::output
