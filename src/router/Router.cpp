#include "router/Router.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::router {

Router::Router(RouterConfig config, std::shared_ptr<RouterBackend> backend)
    : config_(config), backend_(std::move(backend)) {
    config_.validate();
    if (!backend_ || !backend_->available()) {
        throw std::invalid_argument("router requires an available backend");
    }
}

const RouterConfig& Router::config() const noexcept { return config_; }

RouterDecision Router::route(LayerId layerId,
                             tensor::TensorView hiddenState,
                             tensor::TensorView routerWeights) {
    if (hiddenState.device() != backend_->device() ||
        routerWeights.device() != backend_->device()) {
        throw std::invalid_argument("router tensors do not match router backend device");
    }
    return backend_->route(layerId, hiddenState, routerWeights, config_);
}

BatchRouterDecision Router::routeBatch(LayerId layerId,
                                       tensor::TensorView hiddenStates,
                                       tensor::TensorView routerWeights) {
    if (hiddenStates.device() != backend_->device() ||
        routerWeights.device() != backend_->device()) {
        throw std::invalid_argument("router tensors do not match router backend device");
    }
    return backend_->routeBatch(layerId, hiddenStates, routerWeights, config_);
}

} // namespace hypermoe::router
