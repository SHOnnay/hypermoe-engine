#include "transformer/output/LMHead.hpp"

#include "tensor/backend/TensorBackend.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::transformer::output {

LMHead::LMHead(std::shared_ptr<tensor::TensorBackend> backend,
               std::size_t hiddenDimension,
               std::size_t vocabularySize,
               models::TensorLayout weightLayout,
               bool tiedEmbeddings)
    : backend_(std::move(backend)),
      hiddenDimension_(hiddenDimension),
      vocabularySize_(vocabularySize),
      weightLayout_(weightLayout),
      tiedEmbeddings_(tiedEmbeddings) {
    if (!backend_ || !backend_->available() ||
        backend_->device() != tensor::Device::cpu() || hiddenDimension_ == 0 ||
        vocabularySize_ == 0 ||
        (tiedEmbeddings_ && weightLayout_ != models::TensorLayout::OutputInput)) {
        throw std::invalid_argument("LM head configuration is invalid");
    }
}

tensor::Tensor LMHead::execute(tensor::TensorView hiddenStates,
                               tensor::TensorView weights) const {
    [[maybe_unused]] const auto hiddenOwner = hiddenStates.lockOwner();
    [[maybe_unused]] const auto weightOwner = weights.lockOwner();
    const auto expectedWeights = weightLayout_ == models::TensorLayout::InputOutput
        ? tensor::Shape{hiddenDimension_, vocabularySize_}
        : tensor::Shape{vocabularySize_, hiddenDimension_};
    if (!hiddenOwner || !weightOwner || !hiddenStates || !weights ||
        !hiddenStates.isContiguous() || !weights.isContiguous() ||
        hiddenStates.device() != tensor::Device::cpu() ||
        weights.device() != tensor::Device::cpu() ||
        hiddenStates.dtype() != tensor::DType::FP32 ||
        weights.dtype() != tensor::DType::FP32 ||
        hiddenStates.shape().rank() != 2 ||
        hiddenStates.shape().dimensions()[1] != hiddenDimension_ ||
        weights.shape() != expectedWeights) {
        throw std::invalid_argument(
            "LM head requires compatible contiguous CPU FP32 tensors");
    }
    const auto tokenCount = hiddenStates.shape().dimensions()[0];
    auto logits = backend_->allocateTensor(
        {tokenCount, vocabularySize_}, tensor::DType::FP32);
    const auto* hidden = static_cast<const float*>(hiddenStates.data());
    const auto* matrix = static_cast<const float*>(weights.data());
    auto* output = static_cast<float*>(logits.data());
    for (std::size_t token = 0; token < tokenCount; ++token) {
        for (std::size_t vocabulary = 0; vocabulary < vocabularySize_; ++vocabulary) {
            double value{};
            for (std::size_t feature = 0; feature < hiddenDimension_; ++feature) {
                const auto weightIndex =
                    weightLayout_ == models::TensorLayout::InputOutput
                    ? feature * vocabularySize_ + vocabulary
                    : vocabulary * hiddenDimension_ + feature;
                value += static_cast<double>(
                    hidden[token * hiddenDimension_ + feature]) * matrix[weightIndex];
            }
            output[token * vocabularySize_ + vocabulary] = static_cast<float>(value);
        }
    }
    return logits;
}

std::size_t LMHead::vocabularySize() const noexcept { return vocabularySize_; }
std::size_t LMHead::hiddenDimension() const noexcept { return hiddenDimension_; }
models::TensorLayout LMHead::weightLayout() const noexcept { return weightLayout_; }
bool LMHead::tiedEmbeddings() const noexcept { return tiedEmbeddings_; }

} // namespace hypermoe::transformer::output
