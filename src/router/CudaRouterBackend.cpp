#include "router/CudaRouterBackend.hpp"

#include "router/CpuRouterBackend.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::router {
namespace {

tensor::Tensor toHost(tensor::TensorBackend& backend, tensor::TensorView value) {
    [[maybe_unused]] const auto owner = value.lockOwner();
    if (!owner || !value || value.device() != backend.device() ||
        !value.isContiguous()) {
        throw std::invalid_argument("CUDA router received an incompatible tensor");
    }
    tensor::CpuTensorBackend cpu;
    auto result = cpu.allocateTensor(value.shape(), value.dtype());
    backend.copyTensor(value, result.view());
    return result;
}

} // namespace

CudaRouterBackend::CudaRouterBackend(
    std::shared_ptr<tensor::TensorBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available() ||
        backend_->device().type != tensor::DeviceType::CUDA) {
        throw std::invalid_argument("CUDA router requires an available CUDA backend");
    }
}

std::string_view CudaRouterBackend::name() const noexcept {
    return "CUDA host-staged reference router";
}
tensor::Device CudaRouterBackend::device() const noexcept {
    return backend_->device();
}
bool CudaRouterBackend::available() const noexcept { return backend_->available(); }

RouterDecision CudaRouterBackend::route(
    LayerId layerId, tensor::TensorView hiddenState,
    tensor::TensorView routerWeights, const RouterConfig& config) {
    auto hidden = toHost(*backend_, hiddenState);
    auto weights = toHost(*backend_, routerWeights);
    CpuRouterBackend cpu;
    return cpu.route(layerId, hidden.view(), weights.view(), config);
}

BatchRouterDecision CudaRouterBackend::routeBatch(
    LayerId layerId, tensor::TensorView hiddenStates,
    tensor::TensorView routerWeights, const RouterConfig& config) {
    auto hidden = toHost(*backend_, hiddenStates);
    auto weights = toHost(*backend_, routerWeights);
    CpuRouterBackend cpu;
    return cpu.routeBatch(layerId, hidden.view(), weights.view(), config);
}

} // namespace hypermoe::router
