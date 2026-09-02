#include "scheduler/Prefetcher.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace hypermoe::scheduler {

LocalityPrefetcher::LocalityPrefetcher(std::size_t maximumPredictions)
    : maximumPredictions_(maximumPredictions) {
    if (maximumPredictions == 0) {
        throw std::invalid_argument("prefetcher must permit at least one prediction");
    }
}

std::vector<PredictedExpertRequest>
LocalityPrefetcher::predict(const PredictionInput& input) const {
    const auto nextLayer = input.currentLayer + 1U;
    std::unordered_map<ExpertId, double> scores;
    const auto pattern = input.workloadPattern.find(nextLayer);
    if (pattern != input.workloadPattern.end()) {
        for (std::size_t index = 0; index < pattern->second.size(); ++index) {
            const auto positionWeight =
                1.0 - 0.25 * static_cast<double>(index) /
                          static_cast<double>(std::max<std::size_t>(1, pattern->second.size()));
            scores[pattern->second[index]] += 0.6 * positionWeight;
        }
    }
    for (std::size_t index = 0; index < input.recentExperts.size(); ++index) {
        const auto recency = static_cast<double>(index + 1) /
                             static_cast<double>(input.recentExperts.size());
        scores[input.recentExperts[index]] += 0.4 * recency;
    }

    std::vector<PredictedExpertRequest> predictions;
    predictions.reserve(scores.size());
    for (const auto& [id, score] : scores) {
        predictions.push_back({nextLayer, id, std::clamp(score, 0.0, 1.0)});
    }
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& left, const auto& right) {
                  if (left.confidence != right.confidence) {
                      return left.confidence > right.confidence;
                  }
                  return left.expertId < right.expertId;
              });
    if (predictions.size() > maximumPredictions_) {
        predictions.resize(maximumPredictions_);
    }
    return predictions;
}

} // namespace hypermoe::scheduler
