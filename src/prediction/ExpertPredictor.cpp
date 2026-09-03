#include "prediction/ExpertPredictor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace hypermoe::prediction {

ExpertPredictor::ExpertPredictor(std::shared_ptr<TransitionDatabase> database,
                                 std::size_t maximumPredictions,
                                 double minimumConfidence)
    : database_(std::move(database)),
      maximumPredictions_(maximumPredictions),
      minimumConfidence_(minimumConfidence) {
    if (!database_ || maximumPredictions == 0 || !std::isfinite(minimumConfidence) ||
        minimumConfidence < 0.0 || minimumConfidence > 1.0) {
        throw std::invalid_argument("invalid expert predictor configuration");
    }
}

void ExpertPredictor::observe(const router::RouterDecision& decision) {
    database_->record(decision);
}

void ExpertPredictor::observe(const router::RouterDecision& decision,
                              std::uint64_t streamId) {
    database_->record(decision, streamId);
}

std::vector<scheduler::PredictedExpertRequest> ExpertPredictor::predict(
    const scheduler::PredictionInput& input) const {
    if (input.currentLayer == std::numeric_limits<LayerId>::max()) return {};
    const auto targetLayer = static_cast<LayerId>(input.currentLayer + 1U);
    std::vector<ExpertSelection> sources;
    for (const auto expert : input.recentExperts) {
        sources.push_back({input.currentLayer, expert});
    }
    if (sources.empty()) sources = database_->previousSelections(0);
    if (sources.empty()) return {};
    const auto statistics = database_->predictionStatistics(targetLayer, sources);
    if (statistics.candidates.empty()) return {};
    std::unordered_map<ExpertId, double> baseScores;
    for (const auto& candidate : statistics.candidates) {
        double transitionProbability{};
        for (std::size_t source = 0; source < sources.size(); ++source) {
            if (statistics.outgoingBySource[source] != 0) {
                transitionProbability +=
                    static_cast<double>(candidate.incomingBySource[source]) /
                    static_cast<double>(statistics.outgoingBySource[source]);
            }
        }
        transitionProbability /= static_cast<double>(sources.size());
        const auto frequencyProbability = statistics.layerFrequency == 0
            ? 0.0
            : static_cast<double>(candidate.frequency) /
                  static_cast<double>(statistics.layerFrequency);
        const auto recent = std::find(input.recentExperts.begin(), input.recentExperts.end(),
                                      candidate.expert.expertId) != input.recentExperts.end()
                                ? 1.0
                                : 0.0;
        baseScores[candidate.expert.expertId] = 0.7 * transitionProbability +
                                                0.2 * frequencyProbability + 0.1 * recent;
    }
    std::vector<scheduler::PredictedExpertRequest> predictions;
    for (const auto& candidate : statistics.candidates) {
        double cooccurrence{};
        for (const auto& peer : statistics.candidates) {
            if (candidate.expert == peer.expert) continue;
            const auto found = candidate.cooccurrence.find(peer.expert.expertId);
            if (found != candidate.cooccurrence.end()) {
                const auto denominator = std::max<std::uint64_t>(
                    1, std::min(candidate.frequency, peer.frequency));
                cooccurrence = std::max(
                    cooccurrence, static_cast<double>(found->second) /
                                      static_cast<double>(denominator));
            }
        }
        const auto confidence = std::clamp(baseScores.at(candidate.expert.expertId) +
                                               0.05 * cooccurrence,
                                           0.0, 1.0);
        if (confidence >= minimumConfidence_) {
            predictions.push_back({targetLayer, candidate.expert.expertId, confidence});
        }
    }
    std::sort(predictions.begin(), predictions.end(), [](const auto& left, const auto& right) {
        return left.confidence == right.confidence
                   ? left.expertId < right.expertId
                   : left.confidence > right.confidence;
    });
    if (predictions.size() > maximumPredictions_) predictions.resize(maximumPredictions_);
    return predictions;
}

std::vector<scheduler::ScheduleHandle> ExpertPredictor::observeAndPrefetch(
    const router::RouterDecision& decision,
    ExpertHistory& history,
    scheduler::Scheduler& scheduler) {
    history.record(decision);
    observe(decision);
    scheduler::PredictionInput input;
    input.currentLayer = decision.layerId;
    input.recentExperts = decision.selectedExpertIds;
    std::vector<scheduler::ScheduleHandle> handles;
    for (const auto& prediction : predict(input)) {
        try {
            handles.push_back(scheduler.prefetch(prediction));
        } catch (const std::out_of_range&) {
            // A statistically known expert may be absent from a partial runtime graph.
        }
    }
    return handles;
}

const std::shared_ptr<TransitionDatabase>& ExpertPredictor::database() const noexcept {
    return database_;
}

} // namespace hypermoe::prediction
