#include "transformer/embedding/Embedding.hpp"

#include "tensor/backend/TensorBackend.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace hypermoe::transformer::embedding {

Embedding::Embedding(std::shared_ptr<tensor::TensorBackend> backend,
                     std::size_t vocabularySize,
                     std::size_t hiddenDimension)
    : backend_(std::move(backend)),
      vocabularySize_(vocabularySize),
      hiddenDimension_(hiddenDimension) {
    if (!backend_ || !backend_->available() ||
        backend_->device() != tensor::Device::cpu() || vocabularySize_ == 0 ||
        hiddenDimension_ == 0) {
        throw std::invalid_argument(
            "CPU embedding requires an available backend and nonzero dimensions");
    }
}

tensor::Tensor Embedding::execute(std::span<const std::uint32_t> tokenIds,
                                  tensor::TensorView weights) const {
    [[maybe_unused]] const auto owner = weights.lockOwner();
    if (!owner || tokenIds.empty() || !weights || !weights.isContiguous() ||
        weights.device() != device() || weights.dtype() != tensor::DType::FP32 ||
        weights.shape() != tensor::Shape{vocabularySize_, hiddenDimension_}) {
        throw std::invalid_argument(
            "embedding requires a contiguous vocabulary-by-hidden FP32 tensor");
    }
    auto output = backend_->allocateTensor(
        {tokenIds.size(), hiddenDimension_}, tensor::DType::FP32);
    const auto* table = static_cast<const float*>(weights.data());
    auto* destination = static_cast<float*>(output.data());
    for (std::size_t token = 0; token < tokenIds.size(); ++token) {
        if (tokenIds[token] >= vocabularySize_) {
            throw std::out_of_range("token ID exceeds embedding vocabulary");
        }
        std::memcpy(destination + token * hiddenDimension_,
                    table + static_cast<std::size_t>(tokenIds[token]) * hiddenDimension_,
                    hiddenDimension_ * sizeof(float));
    }
    return output;
}

std::size_t Embedding::vocabularySize() const noexcept { return vocabularySize_; }
std::size_t Embedding::hiddenDimension() const noexcept { return hiddenDimension_; }
tensor::Device Embedding::device() const noexcept { return backend_->device(); }

} // namespace hypermoe::transformer::embedding
