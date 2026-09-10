#include "runtime/generation/GenerationState.hpp"

#include <limits>
#include <stdexcept>

namespace hypermoe::runtime::generation {

GenerationState::GenerationState(
    std::size_t maximumNewTokens,
    std::span<const std::uint32_t> stopTokenIds)
    : maximumNewTokens_(maximumNewTokens),
      stopTokenIds_(stopTokenIds.begin(), stopTokenIds.end()) {
    if (maximumNewTokens_ == 0) {
        throw std::invalid_argument("generation requires at least one output token");
    }
}

void GenerationState::begin(std::span<const std::uint32_t> promptTokens) {
    if (active_ || promptTokens.empty()) {
        throw std::logic_error("generation state requires one nonempty prompt");
    }
    promptTokens_.assign(promptTokens.begin(), promptTokens.end());
    generatedTokens_.clear();
    currentTokenPosition_ = 0;
    stopReason_ = GenerationStopReason::None;
    active_ = true;
}

void GenerationState::advancePosition(std::size_t tokenCount) {
    if (!active_ || finished() || tokenCount == 0 ||
        tokenCount > std::numeric_limits<std::uint64_t>::max() -
                         currentTokenPosition_) {
        throw std::logic_error("generation token position cannot advance");
    }
    currentTokenPosition_ += tokenCount;
}

bool GenerationState::appendGenerated(std::uint32_t tokenId) {
    if (!active_ || finished() || generatedTokens_.size() >= maximumNewTokens_) {
        throw std::logic_error("generation state cannot accept another token");
    }
    generatedTokens_.push_back(tokenId);
    if (stopTokenIds_.contains(tokenId)) {
        finish(GenerationStopReason::StopToken);
    } else if (generatedTokens_.size() == maximumNewTokens_) {
        finish(GenerationStopReason::MaximumTokens);
    }
    return finished();
}

void GenerationState::finish(GenerationStopReason reason) {
    if (!active_ || finished() || reason == GenerationStopReason::None) {
        throw std::logic_error("generation stop reason is invalid");
    }
    stopReason_ = reason;
}

std::uint64_t GenerationState::currentTokenPosition() const noexcept {
    return currentTokenPosition_;
}
std::size_t GenerationState::maximumNewTokens() const noexcept {
    return maximumNewTokens_;
}
const std::vector<std::uint32_t>& GenerationState::promptTokens() const noexcept {
    return promptTokens_;
}
const std::vector<std::uint32_t>& GenerationState::generatedTokens() const noexcept {
    return generatedTokens_;
}
bool GenerationState::active() const noexcept { return active_; }
bool GenerationState::finished() const noexcept {
    return stopReason_ != GenerationStopReason::None;
}
GenerationStopReason GenerationState::stopReason() const noexcept {
    return stopReason_;
}

} // namespace hypermoe::runtime::generation
