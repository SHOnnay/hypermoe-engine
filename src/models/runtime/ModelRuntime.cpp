#include "models/runtime/ModelRuntime.hpp"

#include "tensor/backend/TensorBackend.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "transformer/embedding/Embedding.hpp"
#include "transformer/output/FinalNorm.hpp"
#include "transformer/output/LMHead.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::models::runtime {

ModelRuntime::ModelRuntime(const models::ModelManifest& manifest,
                           std::shared_ptr<TransformerModelRuntime> transformer,
                           std::shared_ptr<tensor::TensorBackend> backend)
    : architecture_(ModelArchitecture::fromManifest(manifest)),
      transformer_(std::move(transformer)), backend_(std::move(backend)) {
    if (!manifest.modelIO || !transformer_ || !backend_ ||
        architecture_.vocabularySize == 0 ||
        transformer_->architecture().hiddenDimension !=
            architecture_.hiddenDimension ||
        transformer_->architecture().layerCount != architecture_.layerCount) {
        throw std::invalid_argument(
            "complete model runtime requires compatible architecture and model I/O mappings");
    }
    const auto resolve = [&](std::string_view name,
                             const tensor::Shape& expected) -> tensor::TensorView {
        const auto* metadata = manifest.findTensor(name);
        const auto& value = transformer_->tensors().require(name);
        if (!metadata || metadata->shape != expected || value.shape() != expected ||
            value.dtype() != tensor::DType::FP32 || value.device() != backend_->device()) {
            throw std::invalid_argument(
                "model I/O runtime tensor disagrees with manifest or backend");
        }
        return value.view();
    };
    embeddingWeights_ = resolve(
        manifest.modelIO->tokenEmbeddingTensor,
        {architecture_.vocabularySize, architecture_.hiddenDimension});
    finalNormWeights_ = resolve(
        manifest.modelIO->finalNormTensor, {architecture_.hiddenDimension});
    const auto headShape = manifest.modelIO->lmHead.layout == TensorLayout::InputOutput
        ? tensor::Shape{architecture_.hiddenDimension, architecture_.vocabularySize}
        : tensor::Shape{architecture_.vocabularySize, architecture_.hiddenDimension};
    lmHeadWeights_ = resolve(manifest.modelIO->lmHead.tensorName, headShape);
    embedding_ = std::make_shared<transformer::embedding::Embedding>(
        backend_, architecture_.vocabularySize, architecture_.hiddenDimension);
    finalNorm_ = std::make_shared<transformer::output::FinalNorm>(
        backend_, architecture_.hiddenDimension,
        architecture_.finalNormalization.epsilon);
    lmHead_ = std::make_shared<transformer::output::LMHead>(
        backend_, architecture_.hiddenDimension,
        architecture_.vocabularySize, manifest.modelIO->lmHead.layout,
        manifest.modelIO->tiedEmbeddings);
}

ModelForwardResult ModelRuntime::forward(
    hypermoe::runtime::InferenceContext& context,
    std::span<const std::uint32_t> tokenIds) {
    return forwardImpl(context, tokenIds, nullptr);
}

ModelForwardResult ModelRuntime::forward(
    hypermoe::runtime::InferenceContext& context,
    std::span<const std::uint32_t> tokenIds,
    hypermoe::runtime::cache::KVCacheBase& kvCache) {
    return forwardImpl(context, tokenIds, &kvCache);
}

ModelForwardResult ModelRuntime::forwardImpl(
    hypermoe::runtime::InferenceContext& context,
    std::span<const std::uint32_t> tokenIds,
    hypermoe::runtime::cache::KVCacheBase* kvCache) {
    context.validate();
    if (tokenIds.empty() || tokenIds.size() != context.batchSize ||
        context.hiddenDimension != architecture_.hiddenDimension) {
        throw std::invalid_argument(
            "forward token batch and inference context do not match architecture");
    }
    ModelForwardResult result;
    const auto totalStart = std::chrono::steady_clock::now();
    auto start = totalStart;
    result.embeddings = embedding_->execute(tokenIds, embeddingWeights_);
    result.timings.embedding = std::chrono::steady_clock::now() - start;
    start = std::chrono::steady_clock::now();
    result.transformer = kvCache
        ? transformer_->execute(context, result.embeddings.view(), *kvCache)
        : transformer_->execute(context, result.embeddings.view());
    result.timings.transformer = std::chrono::steady_clock::now() - start;
    start = std::chrono::steady_clock::now();
    result.normalizedHiddenStates = finalNorm_->execute(
        result.transformer.output.view(), finalNormWeights_);
    result.timings.finalNormalization = std::chrono::steady_clock::now() - start;
    start = std::chrono::steady_clock::now();
    result.logits = lmHead_->execute(
        result.normalizedHiddenStates.view(), lmHeadWeights_);
    result.timings.lmHead = std::chrono::steady_clock::now() - start;
    result.timings.total = std::chrono::steady_clock::now() - totalStart;
    return result;
}

const ModelArchitecture& ModelRuntime::architecture() const noexcept {
    return architecture_;
}

tensor::Device ModelRuntime::device() const noexcept {
    return backend_->device();
}

tensor::Tensor ModelRuntime::materializeHost(tensor::TensorView value) const {
    [[maybe_unused]] const auto owner = value.lockOwner();
    if (!owner || !value || value.device() != device() || !value.isContiguous()) {
        throw std::invalid_argument("model runtime cannot materialize incompatible tensor");
    }
    tensor::CpuTensorBackend cpu;
    auto host = cpu.allocateTensor(value.shape(), value.dtype());
    if (device() == tensor::Device::cpu()) {
        cpu.copyTensor(value, host.view());
    } else {
        backend_->copyTensor(value, host.view());
    }
    return host;
}

bool ModelRuntime::tiedWeightsShareStorage() const noexcept {
    return architecture_.tiedEmbeddings &&
           embeddingWeights_.data() == lmHeadWeights_.data();
}

} // namespace hypermoe::models::runtime
