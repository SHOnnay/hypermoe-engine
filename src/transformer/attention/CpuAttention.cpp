#include "transformer/attention/CpuAttention.hpp"

#include "runtime/cache/KVCache.hpp"
#include "tensor/backend/TensorBackend.hpp"
#include "transformer/position/RoPE.hpp"

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

std::size_t checkedWidth(std::size_t heads, std::size_t dimension) {
    if (heads == 0 || dimension == 0 ||
        heads > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::invalid_argument("attention head dimensions are invalid");
    }
    return heads * dimension;
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
    return "CPU reference multi-head attention";
}

tensor::Device CpuAttention::device() const noexcept { return tensor::Device::cpu(); }

AttentionResult CpuAttention::execute(
    tensor::TensorView hiddenStates,
    const AttentionWeights& weights,
    const AttentionConfiguration& requested) {
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
    AttentionConfiguration configuration = requested;
    if (configuration.headDimension == 0) {
        if (configuration.headCount == 0 ||
            queryShape[1] % configuration.headCount != 0) {
            throw std::invalid_argument("query width cannot infer attention head size");
        }
        configuration.headDimension = queryShape[1] / configuration.headCount;
    }
    const auto queryWidth = checkedWidth(configuration.headCount,
                                         configuration.headDimension);
    const auto keyValueWidth = checkedWidth(configuration.keyValueHeadCount,
                                            configuration.headDimension);
    if (configuration.headCount % configuration.keyValueHeadCount != 0 ||
        queryShape[0] != hiddenShape[1] || keyShape[0] != hiddenShape[1] ||
        valueShape[0] != hiddenShape[1] || queryShape[1] != queryWidth ||
        keyShape[1] != keyValueWidth || valueShape[1] != keyValueWidth ||
        outputShape[0] != queryWidth || outputShape[1] != hiddenShape[1]) {
        throw std::invalid_argument("attention projection dimensions are incompatible");
    }
    if (!std::isfinite(configuration.ropeTheta) || configuration.ropeTheta <= 0.0F ||
        configuration.positionOffset >
            std::numeric_limits<std::size_t>::max() ||
        configuration.positionOffset >
            std::numeric_limits<std::uint64_t>::max() - hiddenShape[0]) {
        throw std::invalid_argument("attention position configuration is invalid");
    }

    const auto tokenCount = hiddenShape[0];
    AttentionResult result;
    result.query = backend_->allocateTensor({tokenCount, queryWidth},
                                            tensor::DType::FP32);
    result.key = backend_->allocateTensor({tokenCount, keyValueWidth},
                                          tensor::DType::FP32);
    result.value = backend_->allocateTensor({tokenCount, keyValueWidth},
                                            tensor::DType::FP32);
    backend_->matmul(hiddenStates, weights.query, result.query);
    backend_->matmul(hiddenStates, weights.key, result.key);
    backend_->matmul(hiddenStates, weights.value, result.value);

    if (configuration.rotaryEmbedding) {
        position::RoPE rope(configuration.ropeTheta);
        rope.apply({static_cast<float*>(result.query.data()),
                    result.query.shape().elementCount()},
                   tokenCount, configuration.headCount,
                   configuration.headDimension,
                   static_cast<std::size_t>(configuration.positionOffset));
        rope.apply({static_cast<float*>(result.key.data()),
                    result.key.shape().elementCount()},
                   tokenCount, configuration.keyValueHeadCount,
                   configuration.headDimension,
                   static_cast<std::size_t>(configuration.positionOffset));
    }

    hypermoe::runtime::cache::KVCacheSnapshot cached;
    if (configuration.kvCache) {
        if (configuration.kvCache->keyValueHeads() !=
                configuration.keyValueHeadCount ||
            configuration.kvCache->headDimension() !=
                configuration.headDimension) {
            throw std::invalid_argument("attention and KV cache dimensions disagree");
        }
        auto keyView = result.key.reshape(
            {tokenCount, configuration.keyValueHeadCount,
             configuration.headDimension});
        auto valueView = result.value.reshape(
            {tokenCount, configuration.keyValueHeadCount,
             configuration.headDimension});
        configuration.kvCache->append(
            configuration.layerIndex, configuration.positionOffset,
            keyView, valueView);
        cached = configuration.kvCache->snapshot(configuration.layerIndex);
    } else {
        cached.keyValueHeads = configuration.keyValueHeadCount;
        cached.headDimension = configuration.headDimension;
        const auto* keys = static_cast<const float*>(result.key.data());
        const auto* values = static_cast<const float*>(result.value.data());
        cached.keys.assign(keys, keys + result.key.shape().elementCount());
        cached.values.assign(values, values + result.value.shape().elementCount());
        cached.positions.reserve(tokenCount);
        for (std::size_t token = 0; token < tokenCount; ++token) {
            cached.positions.push_back(configuration.positionOffset + token);
        }
    }
    const auto keyTokenCount = cached.tokenCount();
    result.scores = backend_->allocateTensor(
        {configuration.headCount, tokenCount, keyTokenCount},
        tensor::DType::FP32);
    result.probabilities = backend_->allocateTensor(
        result.scores.shape(), tensor::DType::FP32);
    result.context = backend_->allocateTensor({tokenCount, queryWidth},
                                              tensor::DType::FP32);
    result.output = backend_->allocateTensor({tokenCount, hiddenShape[1]},
                                             tensor::DType::FP32);

    const auto* query = static_cast<const float*>(result.query.data());
    auto* scores = static_cast<float*>(result.scores.data());
    auto* probabilities = static_cast<float*>(result.probabilities.data());
    auto* context = static_cast<float*>(result.context.data());
    std::fill_n(context, result.context.shape().elementCount(), 0.0F);
    const auto headsPerKeyValueHead =
        configuration.headCount / configuration.keyValueHeadCount;
    const auto scale = 1.0 / std::sqrt(
        static_cast<double>(configuration.headDimension));
    for (std::size_t head = 0; head < configuration.headCount; ++head) {
        const auto keyValueHead = head / headsPerKeyValueHead;
        for (std::size_t row = 0; row < tokenCount; ++row) {
            const auto queryPosition = configuration.positionOffset + row;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t column = 0; column < keyTokenCount; ++column) {
                const auto scoreIndex =
                    (head * tokenCount + row) * keyTokenCount + column;
                if (configuration.causal &&
                    cached.positions[column] > queryPosition) {
                    scores[scoreIndex] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                double dot{};
                for (std::size_t feature = 0;
                     feature < configuration.headDimension; ++feature) {
                    const auto queryIndex =
                        row * queryWidth + head * configuration.headDimension + feature;
                    const auto keyIndex =
                        (column * configuration.keyValueHeadCount + keyValueHead) *
                            configuration.headDimension + feature;
                    dot += static_cast<double>(query[queryIndex]) *
                           cached.keys[keyIndex];
                }
                const auto score = static_cast<float>(dot * scale);
                if (!std::isfinite(score)) {
                    throw std::runtime_error("attention produced a non-finite score");
                }
                scores[scoreIndex] = score;
                maximum = std::max(maximum, score);
            }
            double denominator{};
            for (std::size_t column = 0; column < keyTokenCount; ++column) {
                const auto scoreIndex =
                    (head * tokenCount + row) * keyTokenCount + column;
                const auto probability = std::isfinite(scores[scoreIndex])
                    ? std::exp(scores[scoreIndex] - maximum) : 0.0F;
                probabilities[scoreIndex] = probability;
                denominator += probability;
            }
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                throw std::runtime_error("attention softmax normalization failed");
            }
            for (std::size_t column = 0; column < keyTokenCount; ++column) {
                const auto scoreIndex =
                    (head * tokenCount + row) * keyTokenCount + column;
                const auto probability = static_cast<float>(
                    probabilities[scoreIndex] / denominator);
                probabilities[scoreIndex] = probability;
                for (std::size_t feature = 0;
                     feature < configuration.headDimension; ++feature) {
                    const auto valueIndex =
                        (column * configuration.keyValueHeadCount + keyValueHead) *
                            configuration.headDimension + feature;
                    const auto contextIndex =
                        row * queryWidth + head * configuration.headDimension + feature;
                    context[contextIndex] += probability * cached.values[valueIndex];
                }
            }
        }
    }
    backend_->matmul(result.context, weights.output, result.output);
    backend_->synchronize();
    return result;
}

} // namespace hypermoe::transformer::attention
