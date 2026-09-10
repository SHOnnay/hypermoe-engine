#include "runtime/generation/GenerationModel.hpp"

#include "models/runtime/ModelRuntime.hpp"
#include "runtime/cache/KVCache.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace hypermoe::runtime::generation {

tensor::Tensor GenerationModel::materializeHost(tensor::TensorView value) const {
    [[maybe_unused]] const auto owner = value.lockOwner();
    if (!owner || !value || value.device() != tensor::Device::cpu() ||
        !value.isContiguous()) {
        throw std::invalid_argument(
            "generation model cannot materialize incompatible host tensor");
    }
    tensor::CpuTensorBackend cpu;
    auto result = cpu.allocateTensor(value.shape(), value.dtype());
    std::memcpy(result.data(), value.data(), value.bytes());
    return result;
}

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
tensor::Device ModelRuntimeGenerationModel::device() const noexcept {
    return runtime_->device();
}

tensor::Tensor ModelRuntimeGenerationModel::materializeHost(
    tensor::TensorView value) const {
    return runtime_->materializeHost(value);
}

ForwardPass ModelRuntimeGenerationModel::forward(
    InferenceContext& context,
    std::span<const std::uint32_t> tokenIds,
    cache::KVCacheBase& kvCache) {
    auto result = runtime_->forward(context, tokenIds, kvCache);
    if (result.transformer.layers.empty()) {
        throw std::runtime_error("model runtime returned no transformer layers");
    }
    auto routing = result.transformer.layers.back().routing;
    return {std::move(result.normalizedHiddenStates), std::move(result.logits),
            result.transformer.layers.back().layerId, std::move(routing)};
}

} // namespace hypermoe::runtime::generation
