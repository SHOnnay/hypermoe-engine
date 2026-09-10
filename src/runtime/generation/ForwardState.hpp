#pragma once

#include "router/RouterDecision.hpp"
#include "tensor/Tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hypermoe::runtime::generation {

struct AttentionMetadata {
    std::size_t inputTokenCount{};
    std::size_t cachedTokenCount{};
};

struct ForwardPass {
    tensor::Tensor hiddenStates;
    tensor::Tensor logits;
    LayerId finalLayer{};
    std::vector<router::RouterDecision> routing;
};

class ForwardState {
public:
    void update(ForwardPass pass,
                std::uint64_t tokenPosition,
                std::size_t cachedTokenCount);
    void reset() noexcept;

    [[nodiscard]] const tensor::Tensor& hiddenStates() const noexcept;
    [[nodiscard]] const tensor::Tensor& logits() const noexcept;
    [[nodiscard]] LayerId currentLayer() const noexcept;
    [[nodiscard]] std::uint64_t tokenPosition() const noexcept;
    [[nodiscard]] const AttentionMetadata& attention() const noexcept;
    [[nodiscard]] const std::vector<router::RouterDecision>& routing() const noexcept;
    [[nodiscard]] bool valid() const noexcept;

private:
    tensor::Tensor hiddenStates_;
    tensor::Tensor logits_;
    LayerId currentLayer_{};
    std::uint64_t tokenPosition_{};
    AttentionMetadata attention_;
    std::vector<router::RouterDecision> routing_;
};

} // namespace hypermoe::runtime::generation
