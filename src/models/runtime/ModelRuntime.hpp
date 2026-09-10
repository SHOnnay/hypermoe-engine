#pragma once

#include "models/ModelManifest.hpp"
#include "models/runtime/TransformerModelRuntime.hpp"
#include "runtime/InferenceContext.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>

namespace hypermoe::tensor {
class TensorBackend;
}
namespace hypermoe::runtime::cache {
class KVCache;
}
namespace hypermoe::transformer::embedding {
class Embedding;
}
namespace hypermoe::transformer::output {
class FinalNorm;
class LMHead;
}

namespace hypermoe::models::runtime {

struct ModelForwardTimings {
    std::chrono::nanoseconds embedding{};
    std::chrono::nanoseconds transformer{};
    std::chrono::nanoseconds finalNormalization{};
    std::chrono::nanoseconds lmHead{};
    std::chrono::nanoseconds total{};
};

struct ModelForwardResult {
    tensor::Tensor embeddings;
    ModelExecutionResult transformer;
    tensor::Tensor normalizedHiddenStates;
    tensor::Tensor logits;
    ModelForwardTimings timings;
};

class ModelRuntime {
public:
    ModelRuntime(const models::ModelManifest& manifest,
                 std::shared_ptr<TransformerModelRuntime> transformer,
                 std::shared_ptr<tensor::TensorBackend> backend);

    [[nodiscard]] ModelForwardResult forward(
        hypermoe::runtime::InferenceContext& context,
        std::span<const std::uint32_t> tokenIds);
    [[nodiscard]] ModelForwardResult forward(
        hypermoe::runtime::InferenceContext& context,
        std::span<const std::uint32_t> tokenIds,
        hypermoe::runtime::cache::KVCache& kvCache);
    [[nodiscard]] const ModelArchitecture& architecture() const noexcept;
    [[nodiscard]] bool tiedWeightsShareStorage() const noexcept;

private:
    ModelArchitecture architecture_;
    std::shared_ptr<TransformerModelRuntime> transformer_;
    std::shared_ptr<transformer::embedding::Embedding> embedding_;
    std::shared_ptr<transformer::output::FinalNorm> finalNorm_;
    std::shared_ptr<transformer::output::LMHead> lmHead_;
    tensor::TensorView embeddingWeights_;
    tensor::TensorView finalNormWeights_;
    tensor::TensorView lmHeadWeights_;

    [[nodiscard]] ModelForwardResult forwardImpl(
        hypermoe::runtime::InferenceContext& context,
        std::span<const std::uint32_t> tokenIds,
        hypermoe::runtime::cache::KVCache* kvCache);
};

} // namespace hypermoe::models::runtime
