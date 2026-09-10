#include "runtime/generation/InferenceSession.hpp"

#include "runtime/generation/GenerationModel.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::runtime::generation {

InferenceSession::InferenceSession(
    std::shared_ptr<GenerationModel> model,
    std::shared_ptr<cache::KVCacheManager> cacheManager,
    std::size_t maximumNewTokens,
    std::span<const std::uint32_t> stopTokenIds,
    InferenceConfig config)
    : model_(std::move(model)),
      cacheManager_(std::move(cacheManager)),
      generationState_(maximumNewTokens, stopTokenIds) {
    if (!model_ || !cacheManager_ ||
        model_->layerCount() != cacheManager_->layerCount() ||
        model_->keyValueHeads() != cacheManager_->keyValueHeads() ||
        model_->headDimension() != cacheManager_->headDimension() ||
        model_->device() != config.device ||
        model_->device() != cacheManager_->device() ||
        maximumNewTokens > cacheManager_->maximumSequenceLength()) {
        throw std::invalid_argument(
            "inference session model and KV cache manager are incompatible");
    }
    auto allocation = cacheManager_->allocateSession();
    id_ = allocation.sessionId;
    kvCache_ = std::move(allocation.cache);
}

InferenceSession::~InferenceSession() {
    kvCache_.reset();
    if (cacheManager_ && id_ != 0) {
        try {
            cacheManager_->releaseSession(id_);
        } catch (...) {
            // Destruction cannot report an already-released session.
        }
    }
}

GenerationModel& InferenceSession::model() const noexcept { return *model_; }
cache::KVCacheBase& InferenceSession::kvCache() const noexcept { return *kvCache_; }
cache::KVCacheSessionId InferenceSession::id() const noexcept { return id_; }
GenerationState& InferenceSession::generationState() noexcept {
    return generationState_;
}
const GenerationState& InferenceSession::generationState() const noexcept {
    return generationState_;
}
ForwardState& InferenceSession::forwardState() noexcept { return forwardState_; }
const ForwardState& InferenceSession::forwardState() const noexcept {
    return forwardState_;
}

} // namespace hypermoe::runtime::generation
