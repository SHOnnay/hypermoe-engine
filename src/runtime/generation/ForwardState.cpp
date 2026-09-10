#include "runtime/generation/ForwardState.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::runtime::generation {

void ForwardState::update(ForwardPass pass,
                          std::uint64_t tokenPosition,
                          std::size_t cachedTokenCount) {
    if (!pass.hiddenStates || !pass.logits ||
        pass.hiddenStates.shape().rank() != 2 || pass.logits.shape().rank() != 2 ||
        pass.hiddenStates.shape().dimensions()[0] !=
            pass.logits.shape().dimensions()[0] ||
        pass.routing.size() != pass.hiddenStates.shape().dimensions()[0] ||
        pass.hiddenStates.shape().dimensions()[0] >
            std::numeric_limits<std::uint64_t>::max() - tokenPosition) {
        throw std::invalid_argument("forward pass state is inconsistent");
    }
    for (const auto& decision : pass.routing) {
        if (!decision.valid() || decision.layerId != pass.finalLayer) {
            throw std::invalid_argument("forward pass routing metadata is invalid");
        }
    }
    const auto inputTokens = pass.hiddenStates.shape().dimensions()[0];
    hiddenStates_ = std::move(pass.hiddenStates);
    logits_ = std::move(pass.logits);
    currentLayer_ = pass.finalLayer;
    tokenPosition_ = tokenPosition;
    attention_ = {inputTokens, cachedTokenCount};
    routing_ = std::move(pass.routing);
}

void ForwardState::reset() noexcept { *this = {}; }
const tensor::Tensor& ForwardState::hiddenStates() const noexcept {
    return hiddenStates_;
}
const tensor::Tensor& ForwardState::logits() const noexcept { return logits_; }
LayerId ForwardState::currentLayer() const noexcept { return currentLayer_; }
std::uint64_t ForwardState::tokenPosition() const noexcept { return tokenPosition_; }
const AttentionMetadata& ForwardState::attention() const noexcept {
    return attention_;
}
const std::vector<router::RouterDecision>& ForwardState::routing() const noexcept {
    return routing_;
}
bool ForwardState::valid() const noexcept {
    return hiddenStates_.valid() && logits_.valid();
}

} // namespace hypermoe::runtime::generation
