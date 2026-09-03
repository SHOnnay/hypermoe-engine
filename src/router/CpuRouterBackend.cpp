#include "router/CpuRouterBackend.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace hypermoe::router {

std::string_view CpuRouterBackend::name() const noexcept {
    return "CPU reference router";
}

tensor::Device CpuRouterBackend::device() const noexcept {
    return tensor::Device::cpu();
}

bool CpuRouterBackend::available() const noexcept { return true; }

RouterDecision CpuRouterBackend::route(
    LayerId layerId,
    tensor::TensorView hiddenState,
    tensor::TensorView routerWeights,
    const RouterConfig& config) {
    config.validate();
    [[maybe_unused]] const auto hiddenOwner = hiddenState.lockOwner();
    [[maybe_unused]] const auto weightsOwner = routerWeights.lockOwner();
    if (!hiddenOwner || !weightsOwner || !hiddenState || !routerWeights ||
        hiddenState.device() != tensor::Device::cpu() ||
        routerWeights.device() != tensor::Device::cpu() ||
        hiddenState.dtype() != tensor::DType::FP32 ||
        routerWeights.dtype() != tensor::DType::FP32 ||
        !hiddenState.isContiguous() || !routerWeights.isContiguous() ||
        routerWeights.shape().rank() != 2 ||
        (hiddenState.shape().rank() != 1 && hiddenState.shape().rank() != 2)) {
        throw std::invalid_argument(
            "CPU router requires contiguous CPU FP32 hidden state and weights");
    }
    const auto& hiddenDimensions = hiddenState.shape().dimensions();
    if (hiddenState.shape().rank() == 2 && hiddenDimensions[0] != 1) {
        throw std::invalid_argument("CPU router currently routes one token at a time");
    }
    const auto hiddenSize = hiddenDimensions.back();
    const auto& weightDimensions = routerWeights.shape().dimensions();
    if (weightDimensions[0] != hiddenSize ||
        weightDimensions[1] != config.expertCount) {
        throw std::invalid_argument("router weight dimensions do not match configuration");
    }

    const auto* hidden = static_cast<const float*>(hiddenState.data());
    const auto* weights = static_cast<const float*>(routerWeights.data());
    std::vector<float> scores(config.expertCount, 0.0F);
    for (std::size_t expert = 0; expert < config.expertCount; ++expert) {
        double score = 0.0;
        for (std::size_t index = 0; index < hiddenSize; ++index) {
            score += static_cast<double>(hidden[index]) *
                     static_cast<double>(weights[index * config.expertCount + expert]);
        }
        if (!std::isfinite(score)) {
            throw std::runtime_error("router produced a non-finite score");
        }
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
                          const auto leftScore = scores[left];
                          const auto rightScore = scores[right];
                          return leftScore == rightScore ? left < right
                                                        : leftScore > rightScore;
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

} // namespace hypermoe::router
