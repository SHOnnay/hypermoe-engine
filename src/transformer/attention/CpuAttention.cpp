#include "transformer/attention/CpuAttention.hpp"

#include "tensor/backend/TensorBackend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypermoe::transformer::attention {
namespace {

void validateMatrix(tensor::TensorView value, const char* name) {
    if (!value || value.device() != tensor::Device::cpu() ||
        value.dtype() != tensor::DType::FP32 || !value.isContiguous() ||
        value.shape().rank() != 2) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a contiguous CPU FP32 matrix");
    }
}

} // namespace

CpuAttention::CpuAttention(std::shared_ptr<tensor::TensorBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available() ||
        backend_->device() != tensor::Device::cpu()) {
        throw std::invalid_argument("CPU attention requires an available CPU tensor backend");
    }
}

std::string_view CpuAttention::name() const noexcept {
    return "CPU reference scaled dot-product attention";
}

tensor::Device CpuAttention::device() const noexcept { return tensor::Device::cpu(); }

AttentionResult CpuAttention::execute(
    tensor::TensorView hiddenStates,
    const AttentionWeights& weights) {
    [[maybe_unused]] const auto hiddenOwner = hiddenStates.lockOwner();
    [[maybe_unused]] const auto queryOwner = weights.query.lockOwner();
    [[maybe_unused]] const auto keyOwner = weights.key.lockOwner();
    [[maybe_unused]] const auto valueOwner = weights.value.lockOwner();
    [[maybe_unused]] const auto outputOwner = weights.output.lockOwner();
    if (!hiddenOwner || !queryOwner || !keyOwner || !valueOwner || !outputOwner) {
        throw std::invalid_argument("attention received expired tensor storage");
    }
    validateMatrix(hiddenStates, "hidden states");
    validateMatrix(weights.query, "query projection");
    validateMatrix(weights.key, "key projection");
    validateMatrix(weights.value, "value projection");
    validateMatrix(weights.output, "output projection");

    const auto& hiddenShape = hiddenStates.shape().dimensions();
    const auto& queryShape = weights.query.shape().dimensions();
    const auto& keyShape = weights.key.shape().dimensions();
    const auto& valueShape = weights.value.shape().dimensions();
    const auto& outputShape = weights.output.shape().dimensions();
    if (queryShape[0] != hiddenShape[1] || keyShape[0] != hiddenShape[1] ||
        valueShape[0] != hiddenShape[1] || queryShape[1] != keyShape[1] ||
        outputShape[0] != valueShape[1]) {
        throw std::invalid_argument("attention projection dimensions are incompatible");
    }

    const auto tokenCount = hiddenShape[0];
    const auto queryWidth = queryShape[1];
    const auto valueWidth = valueShape[1];
    if (queryWidth == 0 || queryWidth >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("attention query width is invalid");
    }

    AttentionResult result;
    result.query = backend_->allocateTensor({tokenCount, queryWidth},
                                            tensor::DType::FP32);
    result.key = backend_->allocateTensor({tokenCount, queryWidth},
                                          tensor::DType::FP32);
    result.value = backend_->allocateTensor({tokenCount, valueWidth},
                                            tensor::DType::FP32);
    result.scores = backend_->allocateTensor({tokenCount, tokenCount},
                                             tensor::DType::FP32);
    result.probabilities = backend_->allocateTensor({tokenCount, tokenCount},
                                                    tensor::DType::FP32);
    result.context = backend_->allocateTensor({tokenCount, valueWidth},
                                              tensor::DType::FP32);
    result.output = backend_->allocateTensor({tokenCount, outputShape[1]},
                                             tensor::DType::FP32);
    backend_->matmul(hiddenStates, weights.query, result.query);
    backend_->matmul(hiddenStates, weights.key, result.key);
    backend_->matmul(hiddenStates, weights.value, result.value);

    const auto* query = static_cast<const float*>(result.query.data());
    const auto* key = static_cast<const float*>(result.key.data());
    auto* scores = static_cast<float*>(result.scores.data());
    auto* probabilities = static_cast<float*>(result.probabilities.data());
    const auto scale = 1.0 / std::sqrt(static_cast<double>(queryWidth));
    for (std::size_t row = 0; row < tokenCount; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t column = 0; column < tokenCount; ++column) {
            double dot{};
            for (std::size_t feature = 0; feature < queryWidth; ++feature) {
                dot += static_cast<double>(query[row * queryWidth + feature]) *
                       key[column * queryWidth + feature];
            }
            const auto score = static_cast<float>(dot * scale);
            if (!std::isfinite(score)) {
                throw std::runtime_error("attention produced a non-finite score");
            }
            scores[row * tokenCount + column] = score;
            maximum = std::max(maximum, score);
        }
        double denominator{};
        for (std::size_t column = 0; column < tokenCount; ++column) {
            const auto probability =
                std::exp(scores[row * tokenCount + column] - maximum);
            probabilities[row * tokenCount + column] = probability;
            denominator += probability;
        }
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            throw std::runtime_error("attention softmax normalization failed");
        }
        for (std::size_t column = 0; column < tokenCount; ++column) {
            probabilities[row * tokenCount + column] = static_cast<float>(
                probabilities[row * tokenCount + column] / denominator);
        }
    }

    const auto* value = static_cast<const float*>(result.value.data());
    auto* context = static_cast<float*>(result.context.data());
    for (std::size_t row = 0; row < tokenCount; ++row) {
        for (std::size_t feature = 0; feature < valueWidth; ++feature) {
            double sum{};
            for (std::size_t column = 0; column < tokenCount; ++column) {
                sum += static_cast<double>(
                           probabilities[row * tokenCount + column]) *
                       value[column * valueWidth + feature];
            }
            context[row * valueWidth + feature] = static_cast<float>(sum);
        }
    }
    backend_->matmul(result.context, weights.output, result.output);
    backend_->synchronize();
    return result;
}

} // namespace hypermoe::transformer::attention
