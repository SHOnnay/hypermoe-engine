#include "generation/Generator.hpp"

#include "generation/Sampler.hpp"
#include "runtime/cache/KVCacheManager.hpp"
#include "runtime/generation/Decoder.hpp"
#include "runtime/generation/GenerationModel.hpp"
#include "runtime/generation/InferenceSession.hpp"
#include "tokenizer/Tokenizer.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <utility>

namespace hypermoe::generation {
namespace {

std::vector<float> lastTokenLogits(
    runtime::generation::GenerationModel& model,
    const runtime::generation::ForwardState& state,
    std::size_t vocabularySize) {
    const auto& logits = state.logits();
    if (!state.valid() || logits.dtype() != tensor::DType::FP32 ||
        !logits.isContiguous() ||
        logits.shape().rank() != 2 ||
        logits.shape().dimensions()[1] != vocabularySize) {
        throw std::runtime_error("generation model returned incompatible logits");
    }
    auto host = model.materializeHost(logits.view());
    const auto tokenCount = logits.shape().dimensions()[0];
    const auto* values = static_cast<const float*>(host.data());
    return {values + (tokenCount - 1) * vocabularySize,
            values + tokenCount * vocabularySize};
}

} // namespace

double GenerationMetrics::decodeTokensPerSecond() const noexcept {
    if (decodeSteps == 0 || decode.count() <= 0) return 0.0;
    const auto seconds = std::chrono::duration<double>(decode).count();
    return static_cast<double>(decodeSteps) / seconds;
}

Generator::Generator(
    std::shared_ptr<tokenizer::Tokenizer> tokenizer,
    std::shared_ptr<runtime::generation::GenerationModel> model,
    std::shared_ptr<runtime::cache::KVCacheManager> cacheManager)
    : tokenizer_(std::move(tokenizer)),
      model_(std::move(model)),
      cacheManager_(std::move(cacheManager)) {
    if (!tokenizer_ || !model_ || !cacheManager_ ||
        tokenizer_->vocabularySize() != model_->vocabularySize()) {
        throw std::invalid_argument(
            "generator tokenizer, model, and cache manager are incompatible");
    }
}

GenerationResult Generator::generate(
    std::string_view prompt,
    const GenerationConfig& config) const {
    config.sampling.validate(model_->vocabularySize());
    if (config.maximumNewTokens == 0) {
        throw std::invalid_argument("generation requires at least one new token");
    }
    for (const auto token : config.stopTokenIds) {
        if (token >= model_->vocabularySize()) {
            throw std::invalid_argument("generation stop token exceeds vocabulary");
        }
    }
    const auto totalStart = std::chrono::steady_clock::now();
    auto stageStart = totalStart;
    auto promptTokens = tokenizer_->encode(prompt);
    GenerationMetrics metrics;
    metrics.tokenization = std::chrono::steady_clock::now() - stageStart;
    if (promptTokens.empty() ||
        promptTokens.size() > cacheManager_->maximumSequenceLength() ||
        config.maximumNewTokens - 1 > cacheManager_->maximumSequenceLength() -
                                          promptTokens.size()) {
        throw std::invalid_argument(
            "prompt and requested output exceed the bounded KV cache sequence");
    }
    runtime::generation::InferenceSession session(
        model_, cacheManager_, config.maximumNewTokens, config.stopTokenIds,
        config.inference);
    runtime::generation::Decoder decoder;
    stageStart = std::chrono::steady_clock::now();
    decoder.prefill(session, promptTokens);
    metrics.prefill = std::chrono::steady_clock::now() - stageStart;
    metrics.peakKVCacheBytes = cacheManager_->stats().committedBytes;

    Sampler sampler(config.sampling);
    while (!session.generationState().finished()) {
        const auto logits = lastTokenLogits(
            *model_, session.forwardState(), model_->vocabularySize());
        const auto token = sampler.sample(logits);
        if (session.generationState().appendGenerated(token)) break;
        stageStart = std::chrono::steady_clock::now();
        decoder.decode(session, token);
        metrics.decode += std::chrono::steady_clock::now() - stageStart;
        ++metrics.decodeSteps;
        metrics.peakKVCacheBytes = std::max(
            metrics.peakKVCacheBytes,
            cacheManager_->stats().committedBytes);
    }

    GenerationResult result;
    result.promptTokens = promptTokens;
    result.generatedTokens = session.generationState().generatedTokens();
    result.stopReason = session.generationState().stopReason();
    result.text = tokenizer_->decode(result.generatedTokens);
    metrics.total = std::chrono::steady_clock::now() - totalStart;
    result.metrics = metrics;
    return result;
}

} // namespace hypermoe::generation
