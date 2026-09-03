#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <string_view>

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

class Attention {
public:
    virtual ~Attention() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual tensor::Device device() const noexcept = 0;
    [[nodiscard]] virtual AttentionResult execute(
        tensor::TensorView hiddenStates,
        const AttentionWeights& weights) = 0;
};

} // namespace hypermoe::transformer::attention
