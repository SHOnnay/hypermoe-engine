#pragma once

#include "transformer/attention/Attention.hpp"

#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::transformer::attention {

class CpuAttention final : public Attention {
public:
    explicit CpuAttention(std::shared_ptr<tensor::TensorBackend> backend);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] tensor::Device device() const noexcept override;
    [[nodiscard]] AttentionResult execute(
        tensor::TensorView hiddenStates,
        const AttentionWeights& weights,
        const AttentionConfiguration& configuration = {}) override;

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
};

} // namespace hypermoe::transformer::attention
