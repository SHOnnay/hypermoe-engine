#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::transformer::embedding {

// Reference embedding lookup. The public contract is tensor/backend neutral;
// Phase 14 intentionally provides a CPU implementation only.
class Embedding {
public:
    Embedding(std::shared_ptr<tensor::TensorBackend> backend,
              std::size_t vocabularySize,
              std::size_t hiddenDimension);

    [[nodiscard]] tensor::Tensor execute(
        std::span<const std::uint32_t> tokenIds,
        tensor::TensorView weights) const;
    [[nodiscard]] std::size_t vocabularySize() const noexcept;
    [[nodiscard]] std::size_t hiddenDimension() const noexcept;
    [[nodiscard]] tensor::Device device() const noexcept;

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
    std::size_t vocabularySize_{};
    std::size_t hiddenDimension_{};
};

} // namespace hypermoe::transformer::embedding
