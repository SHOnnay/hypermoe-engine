#pragma once

#include "router/RouterBackend.hpp"

namespace hypermoe::router {

class CpuRouterBackend final : public RouterBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] tensor::Device device() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] RouterDecision route(
        LayerId layerId,
        tensor::TensorView hiddenState,
        tensor::TensorView routerWeights,
        const RouterConfig& config) override;
    [[nodiscard]] BatchRouterDecision routeBatch(
        LayerId layerId,
        tensor::TensorView hiddenStates,
        tensor::TensorView routerWeights,
        const RouterConfig& config) override;
};

} // namespace hypermoe::router
