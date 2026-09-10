#include "runtime/generation/GenerationModel.hpp"

#include "models/runtime/ModelRuntime.hpp"
#include "runtime/cache/KVCache.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::runtime::generation {

ModelRuntimeGenerationModel::ModelRuntimeGenerationModel(
    std::shared_ptr<models::runtime::ModelRuntime> runtime)
    : runtime_(std::move(runtime)) {
    if (!runtime_ || runtime_->architecture().vocabularySize == 0) {
        throw std::invalid_argument("generation model requires a complete model runtime");
    }
}

std::size_t ModelRuntimeGenerationModel::vocabularySize() const noexcept {
    return runtime_->architecture().vocabularySize;
}
std::size_t ModelRuntimeGenerationModel::hiddenDimension() const noexcept {
    return runtime_->architecture().hiddenDimension;
}
std::size_t ModelRuntimeGenerationModel::layerCount() const noexcept {
    return runtime_->architecture().layerCount;
}
std::size_t ModelRuntimeGenerationModel::keyValueHeads() const noexcept {
    return runtime_->architecture().keyValueHeads;
}
std::size_t ModelRuntimeGenerationModel::headDimension() const noexcept {
    return runtime_->architecture().headDimension;
}

ForwardPass ModelRuntimeGenerationModel::forward(
    InferenceContext& context,
    std::span<const std::uint32_t> tokenIds,
    cache::KVCache& kvCache) {
    auto result = runtime_->forward(context, tokenIds, kvCache);
    if (result.transformer.layers.empty()) {
        throw std::runtime_error("model runtime returned no transformer layers");
    }
    auto routing = result.transformer.layers.back().routing;
    return {std::move(result.normalizedHiddenStates), std::move(result.logits),
            result.transformer.layers.back().layerId, std::move(routing)};
}

} // namespace hypermoe::runtime::generation
