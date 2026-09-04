#pragma once

#include "models/ModelManifest.hpp"
#include "models/runtime/ModelArchitecture.hpp"
#include "runtime/InferenceContext.hpp"
#include "tensor/Tensor.hpp"
#include "transformer/runtime/TransformerBlock.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hypermoe::runtime::cache {
class KVCache;
}
namespace hypermoe::tensor {
class TensorBackend;
}
namespace hypermoe::transformer {
class MoELayer;
namespace attention {
class Attention;
}
namespace norm {
class Norm;
}
}

namespace hypermoe::models::runtime {

class RuntimeTensorMap {
public:
    void add(std::string name, tensor::Tensor tensor);
    [[nodiscard]] const tensor::Tensor& require(std::string_view name) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t memoryUsageBytes() const noexcept;

private:
    std::unordered_map<std::string, tensor::Tensor> tensors_;
};

struct ModelLayerExecution {
    LayerId layerId{};
    transformer::runtime::TransformerBlockTimings timings;
    hypermoe::runtime::ExecutionMetadata execution;
    std::vector<router::RouterDecision> routing;
    tensor::Tensor output;
};

struct ModelExecutionResult {
    tensor::Tensor output;
    std::vector<ModelLayerExecution> layers;
    std::chrono::nanoseconds totalTime{};
};

class TransformerModelRuntime {
public:
    TransformerModelRuntime(
        const models::ModelManifest& manifest,
        RuntimeTensorMap tensors,
        std::shared_ptr<transformer::attention::Attention> attention,
        std::shared_ptr<transformer::norm::Norm> inputNormalization,
        std::shared_ptr<transformer::norm::Norm> postAttentionNormalization,
        std::shared_ptr<transformer::MoELayer> moe,
        std::shared_ptr<tensor::TensorBackend> backend,
        std::shared_ptr<hypermoe::runtime::cache::KVCache> kvCache = {});

    [[nodiscard]] ModelExecutionResult execute(
        hypermoe::runtime::InferenceContext& context,
        tensor::TensorView hiddenStates);
    [[nodiscard]] const ModelArchitecture& architecture() const noexcept;
    [[nodiscard]] const RuntimeTensorMap& tensors() const noexcept;

private:
    ModelArchitecture architecture_;
    RuntimeTensorMap tensors_;
    std::shared_ptr<transformer::norm::Norm> inputNormalization_;
    std::shared_ptr<transformer::norm::Norm> postAttentionNormalization_;
    std::shared_ptr<tensor::TensorBackend> backend_;
    transformer::runtime::TransformerBlock block_;
    std::shared_ptr<hypermoe::runtime::cache::KVCache> kvCache_;
    std::vector<transformer::runtime::TransformerBlockWeights> weights_;
};

} // namespace hypermoe::models::runtime
