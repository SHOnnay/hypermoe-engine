#include "transformer/norm/RMSNorm.hpp"

#include "tensor/backend/TensorBackend.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace hypermoe::transformer::norm {

RMSNorm::RMSNorm(std::shared_ptr<tensor::TensorBackend> backend,
                 std::size_t hiddenDimension,
                 float epsilon)
    : backend_(std::move(backend)),
      hiddenDimension_(hiddenDimension),
      epsilon_(epsilon) {
    if (!backend_ || !backend_->available() ||
        backend_->device() != tensor::Device::cpu()) {
        throw std::invalid_argument("CPU RMSNorm requires an available CPU backend");
    }
    if (hiddenDimension_ == 0 || !std::isfinite(epsilon_) || epsilon_ <= 0.0F) {
        throw std::invalid_argument("RMSNorm configuration is invalid");
    }
}

std::string_view RMSNorm::name() const noexcept { return "CPU reference RMSNorm"; }
tensor::Device RMSNorm::device() const noexcept { return tensor::Device::cpu(); }
std::size_t RMSNorm::hiddenDimension() const noexcept { return hiddenDimension_; }
float RMSNorm::epsilon() const noexcept { return epsilon_; }

tensor::Tensor RMSNorm::execute(tensor::TensorView input,
                                tensor::TensorView weight) {
    [[maybe_unused]] const auto inputOwner = input.lockOwner();
    [[maybe_unused]] const auto weightOwner = weight.lockOwner();
    if (!inputOwner || !weightOwner || !input || !weight ||
        input.device() != device() || weight.device() != device() ||
        input.dtype() != tensor::DType::FP32 ||
        weight.dtype() != tensor::DType::FP32 || !input.isContiguous() ||
        !weight.isContiguous() || input.shape().rank() != 2 ||
        weight.shape().rank() != 1 ||
        input.shape().dimensions()[1] != hiddenDimension_ ||
        weight.shape().dimensions()[0] != hiddenDimension_) {
        throw std::invalid_argument(
            "RMSNorm requires compatible contiguous CPU FP32 tensors");
    }
    auto output = backend_->allocateTensor(input.shape(), tensor::DType::FP32);
    const auto* source = static_cast<const float*>(input.data());
    const auto* scale = static_cast<const float*>(weight.data());
    auto* destination = static_cast<float*>(output.data());
    const auto tokens = input.shape().dimensions()[0];
    for (std::size_t token = 0; token < tokens; ++token) {
        double squareSum{};
        for (std::size_t hidden = 0; hidden < hiddenDimension_; ++hidden) {
            const auto value = source[token * hiddenDimension_ + hidden];
            squareSum += static_cast<double>(value) * value;
        }
        const auto inverseRms = 1.0 / std::sqrt(
            squareSum / static_cast<double>(hiddenDimension_) + epsilon_);
        for (std::size_t hidden = 0; hidden < hiddenDimension_; ++hidden) {
            destination[token * hiddenDimension_ + hidden] = static_cast<float>(
                source[token * hiddenDimension_ + hidden] * inverseRms * scale[hidden]);
        }
    }
    return output;
}

} // namespace hypermoe::transformer::norm
