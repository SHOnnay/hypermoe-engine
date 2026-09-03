#pragma once

#include "router/RouterConfig.hpp"
#include "router/RouterDecision.hpp"
#include "tensor/TensorView.hpp"

#include <string_view>

namespace hypermoe::router {

class RouterBackend {
public:
    virtual ~RouterBackend() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual tensor::Device device() const noexcept = 0;
    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual RouterDecision route(
        LayerId layerId,
        tensor::TensorView hiddenState,
        tensor::TensorView routerWeights,
        const RouterConfig& config) = 0;
    [[nodiscard]] virtual BatchRouterDecision routeBatch(
        LayerId layerId,
        tensor::TensorView hiddenStates,
        tensor::TensorView routerWeights,
        const RouterConfig& config) = 0;
};

} // namespace hypermoe::router
