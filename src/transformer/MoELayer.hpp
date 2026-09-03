#pragma once

#include "transformer/TransformerLayer.hpp"

#include <memory>

namespace hypermoe::runtime {
class MoERuntime;
}
namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::transformer {

class MoELayer final : public TransformerLayer {
public:
    MoELayer(std::shared_ptr<runtime::MoERuntime> runtime,
             std::shared_ptr<tensor::TensorBackend> backend);

    [[nodiscard]] TransformerLayerResult execute(
        LayerId layerId,
        tensor::TensorView hiddenState,
        tensor::TensorView routerWeights) override;

private:
    std::shared_ptr<runtime::MoERuntime> runtime_;
    std::shared_ptr<tensor::TensorBackend> backend_;
};

} // namespace hypermoe::transformer
