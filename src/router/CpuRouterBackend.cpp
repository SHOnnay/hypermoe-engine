#include "router/CpuRouterBackend.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hypermoe::router {
namespace {

void validateInputs(tensor::TensorView hiddenStates,
                    tensor::TensorView routerWeights,
                    const RouterConfig& config) {
    config.validate();
    if (!hiddenStates || !routerWeights ||
        hiddenStates.device() != tensor::Device::cpu() ||
        routerWeights.device() != tensor::Device::cpu() ||
        hiddenStates.dtype() != tensor::DType::FP32 ||
        routerWeights.dtype() != tensor::DType::FP32 ||
        !hiddenStates.isContiguous() || !routerWeights.isContiguous() ||
        hiddenStates.shape().rank() != 2 || routerWeights.shape().rank() != 2) {
        throw std::invalid_argument(
            "CPU router requires contiguous rank-2 CPU FP32 tensors");
    }
    const auto& hiddenDimensions = hiddenStates.shape().dimensions();
    const auto& weightDimensions = routerWeights.shape().dimensions();
    if (hiddenDimensions[1] != weightDimensions[0] ||
        weightDimensions[1] != config.expertCount) {
        throw std::invalid_argument("router tensor dimensions do not match configuration");
    }
}

RouterDecision routeToken(LayerId layerId,
                          const float* hidden,
                          const float* weights,
                          std::size_t hiddenSize,
                          const RouterConfig& config) {
    std::vector<float> scores(config.expertCount, 0.0F);
    for (std::size_t expert = 0; expert < config.expertCount; ++expert) {
        double score = 0.0;
        for (std::size_t index = 0; index < hiddenSize; ++index) {
            score += static_cast<double>(hidden[index]) *
                     static_cast<double>(weights[index * config.expertCount + expert]);
        }
        if (!std::isfinite(score)) throw std::runtime_error("router produced a non-finite score");
        scores[expert] = static_cast<float>(score);
    }
    if (config.normalization == RoutingNormalization::Softmax) {
        const auto maximum = *std::max_element(scores.begin(), scores.end());
        double denominator = 0.0;
        for (auto& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            throw std::runtime_error("router softmax normalization failed");
        }
        for (auto& score : scores) {
            score = static_cast<float>(static_cast<double>(score) / denominator);
        }
    }
    std::vector<ExpertId> indices(config.expertCount);
    std::iota(indices.begin(), indices.end(), ExpertId{0});
    const auto selectedEnd =
        indices.begin() + static_cast<std::vector<ExpertId>::difference_type>(config.topK);
    std::partial_sort(indices.begin(), selectedEnd, indices.end(),
                      [&](ExpertId left, ExpertId right) {
                          return scores[left] == scores[right] ? left < right
                                                              : scores[left] > scores[right];
                      });
    RouterDecision decision;
    decision.layerId = layerId;
    decision.selectedExpertIds.assign(indices.begin(), selectedEnd);
    decision.routingScores.reserve(config.topK);
    for (const auto expert : decision.selectedExpertIds) {
        decision.routingScores.push_back(scores[expert]);
    }
    if (config.renormalizeSelected) {
        const auto sum = std::accumulate(decision.routingScores.begin(),
                                         decision.routingScores.end(), 0.0);
        if (!std::isfinite(sum) || sum == 0.0) {
            throw std::runtime_error("selected router scores cannot be renormalized");
        }
        for (auto& score : decision.routingScores) {
            score = static_cast<float>(static_cast<double>(score) / sum);
        }
    }
    return decision;
}

} // namespace

std::string_view CpuRouterBackend::name() const noexcept {
    return "CPU reference router";
}

tensor::Device CpuRouterBackend::device() const noexcept { return tensor::Device::cpu(); }
bool CpuRouterBackend::available() const noexcept { return true; }

RouterDecision CpuRouterBackend::route(
    LayerId layerId,
    tensor::TensorView hiddenState,
    tensor::TensorView routerWeights,
    const RouterConfig& config) {
    [[maybe_unused]] const auto hiddenOwner = hiddenState.lockOwner();
    [[maybe_unused]] const auto weightsOwner = routerWeights.lockOwner();
    if (!hiddenState ||
        (hiddenState.shape().rank() != 1 && hiddenState.shape().rank() != 2)) {
        throw std::invalid_argument("CPU router hidden state must have rank one or two");
    }
    if (hiddenState.shape().rank() == 1) {
        hiddenState = hiddenState.reshape({1, hiddenState.shape().dimensions()[0]});
    }
    validateInputs(hiddenState, routerWeights, config);
    if (hiddenState.shape().dimensions()[0] != 1) {
        throw std::invalid_argument("single-token route received multiple tokens");
    }
    return routeToken(layerId, static_cast<const float*>(hiddenState.data()),
                      static_cast<const float*>(routerWeights.data()),
                      hiddenState.shape().dimensions()[1], config);
}

BatchRouterDecision CpuRouterBackend::routeBatch(
    LayerId layerId,
    tensor::TensorView hiddenStates,
    tensor::TensorView routerWeights,
    const RouterConfig& config) {
    [[maybe_unused]] const auto hiddenOwner = hiddenStates.lockOwner();
    [[maybe_unused]] const auto weightsOwner = routerWeights.lockOwner();
    validateInputs(hiddenStates, routerWeights, config);
    const auto tokenCount = hiddenStates.shape().dimensions()[0];
    const auto hiddenSize = hiddenStates.shape().dimensions()[1];
    const auto* hidden = static_cast<const float*>(hiddenStates.data());
    const auto* weights = static_cast<const float*>(routerWeights.data());
    BatchRouterDecision batch;
    batch.layerId = layerId;
    batch.tokens.reserve(tokenCount);
    std::unordered_map<ExpertId, std::size_t> groupPositions;
    for (std::size_t token = 0; token < tokenCount; ++token) {
        auto decision = routeToken(layerId, hidden + token * hiddenSize,
                                   weights, hiddenSize, config);
        for (std::size_t rank = 0; rank < decision.selectedExpertIds.size(); ++rank) {
            const auto expert = decision.selectedExpertIds[rank];
            const auto [found, inserted] =
                groupPositions.emplace(expert, batch.expertGroups.size());
            if (inserted) batch.expertGroups.push_back({expert, {}, {}});
            auto& group = batch.expertGroups[found->second];
            group.tokenIndices.push_back(token);
            group.routingScores.push_back(decision.routingScores[rank]);
        }
        batch.tokens.push_back(std::move(decision));
    }
    std::sort(batch.expertGroups.begin(), batch.expertGroups.end(),
              [](const auto& left, const auto& right) {
                  return left.expertId < right.expertId;
              });
    return batch;
}

} // namespace hypermoe::router
