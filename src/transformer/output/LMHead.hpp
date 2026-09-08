#pragma once

#include "models/ModelManifest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <cstddef>
#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::transformer::output {

class LMHead {
public:
    LMHead(std::shared_ptr<tensor::TensorBackend> backend,
           std::size_t hiddenDimension,
           std::size_t vocabularySize,
           models::TensorLayout weightLayout,
           bool tiedEmbeddings);

    [[nodiscard]] tensor::Tensor execute(tensor::TensorView hiddenStates,
                                         tensor::TensorView weights) const;
    [[nodiscard]] std::size_t vocabularySize() const noexcept;
    [[nodiscard]] std::size_t hiddenDimension() const noexcept;
    [[nodiscard]] models::TensorLayout weightLayout() const noexcept;
    [[nodiscard]] bool tiedEmbeddings() const noexcept;

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
    std::size_t hiddenDimension_{};
    std::size_t vocabularySize_{};
    models::TensorLayout weightLayout_{};
    bool tiedEmbeddings_{};
};

} // namespace hypermoe::transformer::output
