#pragma once

#include "router/RouterBackend.hpp"

#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::router {

// Correctness-first CUDA router. Scores are currently evaluated by the CPU
// reference implementation after explicit device-to-host materialization.
class CudaRouterBackend final : public RouterBackend {
public:
    explicit CudaRouterBackend(std::shared_ptr<tensor::TensorBackend> backend);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] tensor::Device device() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] RouterDecision route(
        LayerId layerId, tensor::TensorView hiddenState,
        tensor::TensorView routerWeights, const RouterConfig& config) override;
    [[nodiscard]] BatchRouterDecision routeBatch(
        LayerId layerId, tensor::TensorView hiddenStates,
        tensor::TensorView routerWeights, const RouterConfig& config) override;

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
};

} // namespace hypermoe::router
