#pragma once

#include "transformer/TransformerLayer.hpp"

#include <memory>

namespace hypermoe::runtime {
class MoERuntime;
struct LayerExecutionResult;
struct BatchLayerExecutionResult;
}
namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::transformer {

class MoELayer final : public TransformerLayer {
public:
    MoELayer(std::shared_ptr<hypermoe::runtime::MoERuntime> runtime,
             std::shared_ptr<tensor::TensorBackend> backend);

    [[nodiscard]] TransformerLayerResult execute(
        LayerId layerId,
        tensor::TensorView hiddenState,
        tensor::TensorView routerWeights) override;
    [[nodiscard]] hypermoe::runtime::LayerExecutionResult executeExperts(
        LayerId layerId,
        tensor::TensorView hiddenState,
        tensor::TensorView routerWeights);
    [[nodiscard]] hypermoe::runtime::BatchLayerExecutionResult executeExpertsBatch(
        LayerId layerId,
        tensor::TensorView hiddenStates,
        tensor::TensorView routerWeights);

private:
    std::shared_ptr<hypermoe::runtime::MoERuntime> runtime_;
    std::shared_ptr<tensor::TensorBackend> backend_;
};

} // namespace hypermoe::transformer
