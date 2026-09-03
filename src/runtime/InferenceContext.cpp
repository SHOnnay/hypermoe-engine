#include "runtime/InferenceContext.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::runtime {

void InferenceContext::validate() const {
    if (batchSize == 0 || hiddenDimension == 0) {
        throw std::invalid_argument("inference context dimensions must be nonzero");
    }
    if (!routingDecisions.empty() && routingDecisions.size() != batchSize) {
        throw std::invalid_argument(
            "inference context routing count does not match batch size");
    }
    for (const auto& decision : routingDecisions) {
        if (!decision.valid() || decision.layerId != layerIndex) {
            throw std::invalid_argument("inference context routing metadata is invalid");
        }
    }
}

void InferenceContext::recordRouting(router::BatchRouterDecision routing,
                                     ExecutionMetadata metadata) {
    if (!routing.valid() || routing.layerId != layerIndex ||
        routing.tokens.size() != batchSize) {
        throw std::invalid_argument("routing result does not match inference context");
    }
    routingDecisions = std::move(routing.tokens);
    execution = std::move(metadata);
    validate();
}

void InferenceContext::advanceLayer(LayerId nextLayer) {
    layerIndex = nextLayer;
    routingDecisions.clear();
    execution = {};
}

} // namespace hypermoe::runtime
