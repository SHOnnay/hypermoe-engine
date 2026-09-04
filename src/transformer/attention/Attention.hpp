#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hypermoe::runtime::cache {
class KVCache;
}

namespace hypermoe::transformer::attention {

struct AttentionWeights {
    tensor::TensorView query;
    tensor::TensorView key;
    tensor::TensorView value;
    tensor::TensorView output;
};

struct AttentionResult {
    tensor::Tensor query;
    tensor::Tensor key;
    tensor::Tensor value;
    tensor::Tensor scores;
    tensor::Tensor probabilities;
    tensor::Tensor context;
    tensor::Tensor output;
};

struct AttentionConfiguration {
    std::size_t headCount{1};
    std::size_t keyValueHeadCount{1};
    std::size_t headDimension{};
    bool causal{};
    bool rotaryEmbedding{};
    float ropeTheta{10000.0F};
    std::uint32_t layerIndex{};
    std::uint64_t positionOffset{};
    hypermoe::runtime::cache::KVCache* kvCache{};
};

class Attention {
public:
    virtual ~Attention() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual tensor::Device device() const noexcept = 0;
    [[nodiscard]] virtual AttentionResult execute(
        tensor::TensorView hiddenStates,
        const AttentionWeights& weights,
        const AttentionConfiguration& configuration = {}) = 0;
};

} // namespace hypermoe::transformer::attention
