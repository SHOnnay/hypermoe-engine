#pragma once

#include "experts/ExpertBatch.hpp"
#include "models/ExpertWeightMap.hpp"
#include "prediction/ExpertHistory.hpp"
#include "prediction/ExpertPredictor.hpp"
#include "router/Router.hpp"
#include "scheduler/Scheduler.hpp"
#include "tensor/Tensor.hpp"
#include "runtime/InferenceContext.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace hypermoe {
class ExpertManager;
class ExpertMlpExecutor;
}

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::runtime {

struct LayerExecutionResult {
    router::RouterDecision routing;
    tensor::Tensor output;
};

struct BatchLayerExecutionResult {
    router::BatchRouterDecision routing;
    std::vector<ExpertBatch> expertBatches;
    std::vector<tensor::Tensor> expertOutputs;
    tensor::Tensor output;
    ExecutionMetadata execution;
};

class MoERuntime {
public:
    MoERuntime(std::shared_ptr<router::Router> router,
               std::shared_ptr<scheduler::Scheduler> scheduler,
               ExpertManager& experts,
               models::ExpertWeightMap weightMap,
               std::shared_ptr<tensor::TensorBackend> tensorBackend,
               std::shared_ptr<ExpertMlpExecutor> executor,
               std::shared_ptr<prediction::ExpertHistory> history = {},
               std::shared_ptr<prediction::ExpertPredictor> predictor = {});

    [[nodiscard]] LayerExecutionResult executeLayer(
        LayerId layerId,
        tensor::TensorView hiddenState,
        tensor::TensorView routerWeights);
    [[nodiscard]] BatchLayerExecutionResult executeBatch(
        LayerId layerId,
        tensor::TensorView hiddenStates,
        tensor::TensorView routerWeights);

private:
    [[nodiscard]] static constexpr std::uint64_t key(LayerId layerId,
                                                      ExpertId expertId) noexcept {
        return (static_cast<std::uint64_t>(layerId) << 32U) | expertId;
    }
    std::shared_ptr<router::Router> router_;
    std::shared_ptr<scheduler::Scheduler> scheduler_;
    ExpertManager& experts_;
    models::ExpertWeightMap weightMap_;
    std::shared_ptr<tensor::TensorBackend> tensorBackend_;
    std::shared_ptr<ExpertMlpExecutor> executor_;
    std::shared_ptr<prediction::ExpertHistory> history_;
    std::shared_ptr<prediction::ExpertPredictor> predictor_;
    std::unordered_map<std::uint64_t, std::uint64_t> payloadOffsets_;
    std::mutex executionMutex_;
};

} // namespace hypermoe::runtime
