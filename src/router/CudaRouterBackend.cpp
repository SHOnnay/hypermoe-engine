#include "router/CudaRouterBackend.hpp"

#include "router/CpuRouterBackend.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hypermoe::router {
namespace {

tensor::Tensor toHost(tensor::TensorBackend& backend, tensor::TensorView value) {
    [[maybe_unused]] const auto owner = value.lockOwner();
    if (!owner || !value || value.device() != backend.device() ||
        !value.isContiguous()) {
        throw std::invalid_argument("CUDA router received an incompatible tensor");
    }
    tensor::CpuTensorBackend cpu;
    auto result = cpu.allocateTensor(value.shape(), value.dtype());
    backend.copyTensor(value, result.view());
    return result;
}

BatchRouterDecision makeDecision(
    LayerId layerId, std::size_t tokenCount, const RouterConfig& config,
    const tensor::CudaTensorBackend::RoutingSelection& selection) {
    if (tokenCount > std::numeric_limits<std::size_t>::max() / config.topK) {
        throw std::overflow_error("CUDA router selection size overflows");
    }
    const auto expected = tokenCount * config.topK;
    if (selection.expertIds.size() != expected ||
        selection.scores.size() != selection.expertIds.size()) {
        throw std::runtime_error("CUDA router returned inconsistent selection data");
    }
    BatchRouterDecision batch;
    batch.layerId = layerId;
    batch.tokens.reserve(tokenCount);
    std::unordered_map<ExpertId, std::size_t> positions;
    for (std::size_t token = 0; token < tokenCount; ++token) {
        RouterDecision decision;
        decision.layerId = layerId;
        for (std::size_t rank = 0; rank < config.topK; ++rank) {
            const auto offset = token * config.topK + rank;
            const auto expert = selection.expertIds[offset];
            if (expert >= config.expertCount) {
                throw std::runtime_error("CUDA router selected an invalid expert");
            }
            decision.selectedExpertIds.push_back(expert);
            decision.routingScores.push_back(selection.scores[offset]);
            const auto [entry, inserted] =
                positions.emplace(expert, batch.expertGroups.size());
            if (inserted) batch.expertGroups.push_back({expert, {}, {}});
            auto& group = batch.expertGroups[entry->second];
            group.tokenIndices.push_back(token);
            group.routingScores.push_back(selection.scores[offset]);
        }
        batch.tokens.push_back(std::move(decision));
    }
    std::sort(batch.expertGroups.begin(), batch.expertGroups.end(),
              [](const auto& left, const auto& right) {
                  return left.expertId < right.expertId;
              });
    return batch;
}

} // namespace

CudaRouterBackend::CudaRouterBackend(
    std::shared_ptr<tensor::TensorBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available() ||
        backend_->device().type != tensor::DeviceType::CUDA) {
        throw std::invalid_argument("CUDA router requires an available CUDA backend");
    }
}

std::string_view CudaRouterBackend::name() const noexcept {
    return "CUDA router backend";
}
tensor::Device CudaRouterBackend::device() const noexcept {
    return backend_->device();
}
bool CudaRouterBackend::available() const noexcept { return backend_->available(); }

RouterDecision CudaRouterBackend::route(
    LayerId layerId, tensor::TensorView hiddenState,
    tensor::TensorView routerWeights, const RouterConfig& config) {
    if (hiddenState.shape().rank() == 1) {
        hiddenState = hiddenState.reshape(
            {1, hiddenState.shape().dimensions()[0]});
    }
    auto batch = routeBatch(layerId, hiddenState, routerWeights, config);
    if (batch.tokens.size() != 1) {
        throw std::invalid_argument("single-token CUDA route received multiple tokens");
    }
    return std::move(batch.tokens.front());
}

BatchRouterDecision CudaRouterBackend::routeBatch(
    LayerId layerId, tensor::TensorView hiddenStates,
    tensor::TensorView routerWeights, const RouterConfig& config) {
    config.validate();
    if (auto* cuda = dynamic_cast<tensor::CudaTensorBackend*>(backend_.get());
        cuda && cuda->nativeKernelsAvailable()) {
        const auto selection = cuda->routeTopK(
            hiddenStates, routerWeights, config.expertCount, config.topK,
            config.normalization == RoutingNormalization::Softmax,
            config.renormalizeSelected);
        return makeDecision(layerId, hiddenStates.shape().dimensions()[0],
                            config, selection);
    }
    auto hidden = toHost(*backend_, hiddenStates);
    auto weights = toHost(*backend_, routerWeights);
    CpuRouterBackend cpu;
    return cpu.routeBatch(layerId, hidden.view(), weights.view(), config);
}

} // namespace hypermoe::router
