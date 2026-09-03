#pragma once

#include "core/runtime/MoERuntime.hpp"
#include "runtime/InferenceContext.hpp"
#include "transformer/attention/Attention.hpp"

#include <chrono>
#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}
namespace hypermoe::transformer {
class MoELayer;
namespace norm {
class Norm;
}
}

namespace hypermoe::transformer::runtime {

struct TransformerBlockWeights {
    attention::AttentionWeights attention;
    tensor::TensorView norm;
    tensor::TensorView router;
};

struct TransformerBlockTimings {
    std::chrono::nanoseconds attention{};
    std::chrono::nanoseconds normalization{};
    std::chrono::nanoseconds moe{};
    std::chrono::nanoseconds residual{};
    std::chrono::nanoseconds total{};
};

struct TransformerBlockResult {
    attention::AttentionResult attention;
    tensor::Tensor normalized;
    hypermoe::runtime::BatchLayerExecutionResult moe;
    tensor::Tensor output;
    TransformerBlockTimings timings;
};

class TransformerBlock {
public:
    TransformerBlock(std::shared_ptr<attention::Attention> attention,
                     std::shared_ptr<norm::Norm> normalization,
                     std::shared_ptr<MoELayer> moe,
                     std::shared_ptr<tensor::TensorBackend> backend);

    [[nodiscard]] TransformerBlockResult execute(
        hypermoe::runtime::InferenceContext& context,
        tensor::TensorView hiddenStates,
        const TransformerBlockWeights& weights);

private:
    std::shared_ptr<attention::Attention> attention_;
    std::shared_ptr<norm::Norm> normalization_;
    std::shared_ptr<MoELayer> moe_;
    std::shared_ptr<tensor::TensorBackend> backend_;
};

} // namespace hypermoe::transformer::runtime
