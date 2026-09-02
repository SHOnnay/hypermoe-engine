#include "experts/ExpertExecutor.hpp"

#include "tensor/backend/TensorBackend.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe {

MatmulExpertExecutor::MatmulExpertExecutor(
    std::shared_ptr<tensor::TensorBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available()) {
        throw std::invalid_argument("expert executor requires an available tensor backend");
    }
}

void MatmulExpertExecutor::execute(const tensor::Tensor& input,
                                   const tensor::Tensor& expertWeights,
                                   tensor::Tensor& output) {
    backend_->matmul(input, expertWeights, output);
}

} // namespace hypermoe
