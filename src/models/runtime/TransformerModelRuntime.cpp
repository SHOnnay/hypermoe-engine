#include "models/runtime/TransformerModelRuntime.hpp"

#include "runtime/cache/KVCache.hpp"
#include "tensor/backend/TensorBackend.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/Attention.hpp"
#include "transformer/norm/Norm.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace hypermoe::models::runtime {

void RuntimeTensorMap::add(std::string name, tensor::Tensor value) {
    if (name.empty() || !value || !tensors_.emplace(std::move(name),
                                                     std::move(value)).second) {
        throw std::invalid_argument("runtime tensor mapping is invalid or duplicated");
    }
}

const tensor::Tensor& RuntimeTensorMap::require(std::string_view name) const {
    const auto found = tensors_.find(std::string(name));
    if (found == tensors_.end()) {
        throw std::out_of_range("required runtime tensor is not bound: " +
                                std::string(name));
    }
    return found->second;
}

std::size_t RuntimeTensorMap::size() const noexcept { return tensors_.size(); }

std::size_t RuntimeTensorMap::memoryUsageBytes() const noexcept {
    std::size_t result{};
    for (const auto& [name, value] : tensors_) {
        (void)name;
        result += value.storageBytes();
    }
    return result;
}

TransformerModelRuntime::TransformerModelRuntime(
    const models::ModelManifest& manifest,
    RuntimeTensorMap tensors,
    std::shared_ptr<transformer::attention::Attention> attention,
    std::shared_ptr<transformer::norm::Norm> inputNormalization,
    std::shared_ptr<transformer::norm::Norm> postAttentionNormalization,
    std::shared_ptr<transformer::MoELayer> moe,
    std::shared_ptr<tensor::TensorBackend> backend,
    std::shared_ptr<hypermoe::runtime::cache::KVCacheBase> kvCache)
    : architecture_(ModelArchitecture::fromManifest(manifest)),
      tensors_(std::move(tensors)),
      inputNormalization_(std::move(inputNormalization)),
      postAttentionNormalization_(std::move(postAttentionNormalization)),
      backend_(std::move(backend)),
      block_(std::move(attention), inputNormalization_,
             postAttentionNormalization_, std::move(moe),
             backend_),
      kvCache_(std::move(kvCache)) {
    if (manifest.layers.size() != architecture_.layerCount) {
        throw std::invalid_argument(
            "model runtime requires complete transformer layer mappings");
    }
    if (kvCache_ &&
        (kvCache_->layerCount() != architecture_.layerCount ||
         kvCache_->keyValueHeads() != architecture_.keyValueHeads ||
         kvCache_->headDimension() != architecture_.headDimension ||
         kvCache_->device() != backend_->device())) {
        throw std::invalid_argument("model architecture and KV cache disagree");
    }
    if (!inputNormalization_ || !postAttentionNormalization_ ||
        inputNormalization_->hiddenDimension() != architecture_.hiddenDimension ||
        postAttentionNormalization_->hiddenDimension() != architecture_.hiddenDimension ||
        std::abs(inputNormalization_->epsilon() -
                 architecture_.inputNormalization.epsilon) > 1.0e-12F ||
        std::abs(postAttentionNormalization_->epsilon() -
                 architecture_.postAttentionNormalization.epsilon) > 1.0e-12F) {
        throw std::invalid_argument(
            "normalization implementation disagrees with model architecture");
    }
    weights_.reserve(architecture_.layerCount);
    const auto resolve = [&](std::string_view name) -> const tensor::Tensor& {
        const auto* metadata = manifest.findTensor(name);
        const auto& value = tensors_.require(name);
        if (!metadata || value.shape() != metadata->shape ||
            value.dtype() != tensor::DType::FP32 ||
            !backend_ || value.device() != backend_->device()) {
            throw std::invalid_argument(
                "runtime tensor binding disagrees with manifest or backend");
        }
        return value;
    };
    for (std::size_t layerId = 0; layerId < architecture_.layerCount; ++layerId) {
        const auto* mapping = manifest.findLayer(static_cast<std::uint32_t>(layerId));
        if (!mapping ||
            mapping->queryProjection.layout != TensorLayout::InputOutput ||
            mapping->keyProjection.layout != TensorLayout::InputOutput ||
            mapping->valueProjection.layout != TensorLayout::InputOutput ||
            mapping->outputProjection.layout != TensorLayout::InputOutput) {
            throw std::invalid_argument(
                "runtime transformer projections must use INPUT_OUTPUT layout");
        }
        transformer::attention::AttentionConfiguration attentionConfiguration;
        attentionConfiguration.headCount = architecture_.attentionHeads;
        attentionConfiguration.keyValueHeadCount = architecture_.keyValueHeads;
        attentionConfiguration.headDimension = architecture_.headDimension;
        attentionConfiguration.causal = true;
        attentionConfiguration.rotaryEmbedding = true;
        attentionConfiguration.ropeTheta = architecture_.ropeTheta;
        attentionConfiguration.queryKeyNormEpsilon =
            architecture_.inputNormalization.epsilon;
        attentionConfiguration.layerIndex = static_cast<std::uint32_t>(layerId);
        attentionConfiguration.kvCache = kvCache_.get();
        if (mapping->queryNormTensor.empty() != mapping->keyNormTensor.empty()) {
            throw std::invalid_argument(
                "transformer layer must bind both query and key normalization weights");
        }
        const auto optional = [&](const std::string& name) {
            return name.empty() ? tensor::TensorView{} : resolve(name).view();
        };
        weights_.push_back({
            {resolve(mapping->queryProjection.tensorName).view(),
             resolve(mapping->keyProjection.tensorName).view(),
             resolve(mapping->valueProjection.tensorName).view(),
             resolve(mapping->outputProjection.tensorName).view(),
             optional(mapping->queryNormTensor),
             optional(mapping->keyNormTensor)},
            resolve(mapping->postAttentionNormTensor).view(),
            resolve(mapping->routerTensor).view(),
            attentionConfiguration,
            resolve(mapping->inputNormTensor).view()});
    }
}

ModelExecutionResult TransformerModelRuntime::execute(
    hypermoe::runtime::InferenceContext& context,
    tensor::TensorView hiddenStates) {
    return executeImpl(context, hiddenStates, kvCache_.get());
}

ModelExecutionResult TransformerModelRuntime::execute(
    hypermoe::runtime::InferenceContext& context,
    tensor::TensorView hiddenStates,
    hypermoe::runtime::cache::KVCacheBase& kvCache) {
    return executeImpl(context, hiddenStates, &kvCache);
}

ModelExecutionResult TransformerModelRuntime::executeImpl(
    hypermoe::runtime::InferenceContext& context,
    tensor::TensorView hiddenStates,
    hypermoe::runtime::cache::KVCacheBase* kvCache) {
    context.validate();
    if (!hiddenStates || hiddenStates.shape().rank() != 2 ||
        hiddenStates.shape().dimensions()[0] != context.batchSize ||
        hiddenStates.shape().dimensions()[1] != architecture_.hiddenDimension ||
        context.hiddenDimension != architecture_.hiddenDimension) {
        throw std::invalid_argument(
            "model input and inference context do not match architecture");
    }
    if (kvCache &&
        (kvCache->layerCount() != architecture_.layerCount ||
         kvCache->keyValueHeads() != architecture_.keyValueHeads ||
         kvCache->headDimension() != architecture_.headDimension ||
         kvCache->device() != backend_->device())) {
        throw std::invalid_argument("model architecture and session KV cache disagree");
    }
    ModelExecutionResult modelResult;
    modelResult.layers.reserve(architecture_.layerCount);
    tensor::Tensor current;
    auto currentView = hiddenStates;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t layerId = 0; layerId < architecture_.layerCount; ++layerId) {
        context.advanceLayer(static_cast<LayerId>(layerId));
        auto layerWeights = weights_[layerId];
        layerWeights.attentionConfiguration.kvCache = kvCache;
        auto layerResult = block_.execute(context, currentView, layerWeights);
        modelResult.layers.push_back({
            static_cast<LayerId>(layerId), layerResult.timings,
            layerResult.moe.execution, layerResult.moe.routing.tokens,
            layerResult.attention.output, layerResult.moe.expertOutputs,
            layerResult.output});
        current = std::move(layerResult.output);
        currentView = current.view();
    }
    modelResult.totalTime = std::chrono::steady_clock::now() - start;
    modelResult.output = std::move(current);
    return modelResult;
}

const ModelArchitecture& TransformerModelRuntime::architecture() const noexcept {
    return architecture_;
}

const RuntimeTensorMap& TransformerModelRuntime::tensors() const noexcept {
    return tensors_;
}

} // namespace hypermoe::models::runtime
