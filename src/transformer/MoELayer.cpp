#include "transformer/MoELayer.hpp"

#include "core/runtime/MoERuntime.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::transformer {

MoELayer::MoELayer(std::shared_ptr<runtime::MoERuntime> runtime,
                   std::shared_ptr<tensor::TensorBackend> backend)
    : runtime_(std::move(runtime)), backend_(std::move(backend)) {
    if (!runtime_ || !backend_ || !backend_->available()) {
        throw std::invalid_argument("MoE layer dependencies must be available");
    }
}

TransformerLayerResult MoELayer::execute(
    LayerId layerId,
    tensor::TensorView hiddenState,
    tensor::TensorView routerWeights) {
    if (!hiddenState || hiddenState.shape().rank() != 2 ||
        hiddenState.dtype() != tensor::DType::FP32 ||
        hiddenState.device() != backend_->device()) {
        throw std::invalid_argument(
            "MoE transformer layer requires a rank-2 FP32 hidden state");
    }
    // Attention is an explicit identity placeholder until attention metadata and
    // kernels are introduced. Copying keeps ownership and residual inputs clear.
    auto attention = backend_->allocateTensor(hiddenState.shape(), tensor::DType::FP32);
    backend_->copyTensor(hiddenState, attention);
    auto moe = runtime_->executeLayer(layerId, attention, routerWeights);
    if (moe.output.shape() != attention.shape()) {
        throw std::invalid_argument("MoE output shape is incompatible with residual");
    }
    auto output = backend_->allocateTensor(attention.shape(), tensor::DType::FP32);
    backend_->add(attention, moe.output, output);
    backend_->synchronize();
    return {std::move(moe.routing), std::move(attention), std::move(output)};
}

} // namespace hypermoe::transformer
