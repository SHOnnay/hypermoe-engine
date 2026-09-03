#include "transformer/runtime/TransformerBlock.hpp"

#include "tensor/backend/TensorBackend.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/norm/Norm.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace hypermoe::transformer::runtime {

TransformerBlock::TransformerBlock(
    std::shared_ptr<attention::Attention> attention,
    std::shared_ptr<norm::Norm> normalization,
    std::shared_ptr<MoELayer> moe,
    std::shared_ptr<tensor::TensorBackend> backend)
    : attention_(std::move(attention)),
      normalization_(std::move(normalization)),
      moe_(std::move(moe)),
      backend_(std::move(backend)) {
    if (!attention_ || !normalization_ || !moe_ || !backend_ ||
        !backend_->available() || attention_->device() != backend_->device() ||
        normalization_->device() != backend_->device()) {
        throw std::invalid_argument(
            "transformer block dependencies must use one available backend");
    }
}

TransformerBlockResult TransformerBlock::execute(
    hypermoe::runtime::InferenceContext& context,
    tensor::TensorView hiddenStates,
    const TransformerBlockWeights& weights) {
    context.validate();
    if (!hiddenStates || hiddenStates.device() != backend_->device() ||
        hiddenStates.dtype() != tensor::DType::FP32 ||
        hiddenStates.shape().rank() != 2 || !hiddenStates.isContiguous() ||
        hiddenStates.shape().dimensions()[0] != context.batchSize ||
        hiddenStates.shape().dimensions()[1] != context.hiddenDimension) {
        throw std::invalid_argument(
            "transformer block hidden states do not match inference context");
    }

    TransformerBlockTimings timings;
    const auto totalStart = std::chrono::steady_clock::now();
    auto stageStart = totalStart;
    auto attentionResult = attention_->execute(hiddenStates, weights.attention);
    timings.attention = std::chrono::steady_clock::now() - stageStart;
    if (attentionResult.output.shape() != hiddenStates.shape()) {
        throw std::invalid_argument(
            "attention output must preserve transformer hidden shape");
    }

    stageStart = std::chrono::steady_clock::now();
    auto normalized = normalization_->execute(attentionResult.output, weights.norm);
    timings.normalization = std::chrono::steady_clock::now() - stageStart;

    stageStart = std::chrono::steady_clock::now();
    auto moeResult = moe_->executeExpertsBatch(
        context.layerIndex, normalized, weights.router);
    timings.moe = std::chrono::steady_clock::now() - stageStart;
    if (moeResult.output.shape() != attentionResult.output.shape()) {
        throw std::invalid_argument("MoE output is incompatible with residual input");
    }

    stageStart = std::chrono::steady_clock::now();
    auto output = backend_->allocateTensor(attentionResult.output.shape(),
                                           tensor::DType::FP32);
    backend_->add(attentionResult.output, moeResult.output, output);
    backend_->synchronize();
    timings.residual = std::chrono::steady_clock::now() - stageStart;
    timings.total = std::chrono::steady_clock::now() - totalStart;

    context.recordRouting(moeResult.routing, moeResult.execution);
    return {std::move(attentionResult), std::move(normalized),
            std::move(moeResult), std::move(output), timings};
}

} // namespace hypermoe::transformer::runtime
