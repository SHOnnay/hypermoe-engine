#pragma once

#include "router/RouterBackend.hpp"

#include <memory>

namespace hypermoe::router {

class Router {
public:
    Router(RouterConfig config, std::shared_ptr<RouterBackend> backend);

    [[nodiscard]] const RouterConfig& config() const noexcept;
    [[nodiscard]] RouterDecision route(LayerId layerId,
                                       tensor::TensorView hiddenState,
                                       tensor::TensorView routerWeights);
    [[nodiscard]] BatchRouterDecision routeBatch(
        LayerId layerId,
        tensor::TensorView hiddenStates,
        tensor::TensorView routerWeights);

private:
    RouterConfig config_;
    std::shared_ptr<RouterBackend> backend_;
};

} // namespace hypermoe::router
