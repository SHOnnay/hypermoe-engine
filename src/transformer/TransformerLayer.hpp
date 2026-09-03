#pragma once

#include "router/RouterDecision.hpp"
#include "tensor/Tensor.hpp"

namespace hypermoe::transformer {

struct TransformerLayerResult {
    router::RouterDecision routing;
    tensor::Tensor attentionOutput;
    tensor::Tensor output;
};

class TransformerLayer {
public:
    virtual ~TransformerLayer() = default;
    [[nodiscard]] virtual TransformerLayerResult execute(
        LayerId layerId,
        tensor::TensorView hiddenState,
        tensor::TensorView routerWeights) = 0;
};

} // namespace hypermoe::transformer
