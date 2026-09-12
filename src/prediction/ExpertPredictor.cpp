#include "prediction/ExpertPredictor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hypermoe::prediction {
namespace {

double ratio(double numerator, double denominator) noexcept {
    return denominator <= 0.0 ? 0.0 : numerator / denominator;
}

bool contains(std::span<const ExpertId> experts, ExpertId candidate) {
    return std::find(experts.begin(), experts.end(), candidate) != experts.end();
}

} // namespace

double PredictionQualitySnapshot::top1Accuracy() const noexcept {
    return opportunities == 0
        ? 0.0
        : static_cast<double>(top1Correct) / static_cast<double>(opportunities);
}

double PredictionQualitySnapshot::topKCoverage() const noexcept {
    return opportunities == 0
        ? 0.0
        : static_cast<double>(topKCovered) / static_cast<double>(opportunities);
}

double PredictionQualitySnapshot::prefetchUsefulness() const noexcept {
    return predictions == 0
        ? 0.0
        : static_cast<double>(usefulPredictions) / static_cast<double>(predictions);
}

ExpertPredictor::ExpertPredictor(std::shared_ptr<TransitionDatabase> database,
                                 std::size_t maximumPredictions,
                                 double minimumConfidence)
    : ExpertPredictor(std::move(database),
                      AdaptivePredictionConfig{maximumPredictions,
                                               minimumConfidence,
                                               minimumConfidence}) {}

ExpertPredictor::ExpertPredictor(std::shared_ptr<TransitionDatabase> database,
                                 AdaptivePredictionConfig config,
                                 std::shared_ptr<Profiler> profiler)
    : database_(std::move(database)), config_(config), profiler_(std::move(profiler)) {
    if (!database_ || config_.maximumPredictions == 0 ||
        !std::isfinite(config_.minimumConfidence) ||
        !std::isfinite(config_.minimumPrefetchConfidence) ||
        config_.minimumConfidence < 0.0 || config_.minimumConfidence > 1.0 ||
        config_.minimumPrefetchConfidence < 0.0 ||
        config_.minimumPrefetchConfidence > 1.0) {
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

std::vector<ExpertPrediction> ExpertPredictor::predictDetailed(
    const scheduler::PredictionInput& input) const {
    if (input.currentLayer == std::numeric_limits<LayerId>::max()) return {};
    const auto targetLayer = static_cast<LayerId>(input.currentLayer + 1U);
    std::vector<ExpertSelection> sources;
    sources.reserve(input.recentExperts.size());
    for (const auto expert : input.recentExperts) {
        sources.push_back({input.currentLayer, expert});
    }
    if (sources.empty()) sources = database_->previousSelections(0);
    if (sources.empty()) return {};

    const auto statistics = database_->predictionStatistics(targetLayer, sources);
    if (statistics.candidates.empty()) return {};

    struct CandidateScore {
        ExpertId expertId{};
        double raw{};
        double confidence{};
    };
    std::vector<CandidateScore> scores;
    scores.reserve(statistics.candidates.size());
    double totalRaw{};
    for (const auto& candidate : statistics.candidates) {
        double transitionProbability{};
        std::size_t supportedSources{};
        for (std::size_t source = 0; source < sources.size(); ++source) {
            double outgoing{};
            for (const auto& peer : statistics.candidates) {
                outgoing += peer.decayedIncomingBySource[source];
            }
            if (outgoing > 0.0) {
                transitionProbability +=
                    candidate.decayedIncomingBySource[source] / outgoing;
                if (candidate.decayedIncomingBySource[source] > 0.0) ++supportedSources;
            }
        }
        transitionProbability /= static_cast<double>(sources.size());
        const auto popularity = ratio(candidate.decayedFrequency,
                                      statistics.decayedLayerFrequency);
        const auto windowPopularity = ratio(
            static_cast<double>(candidate.windowFrequency),
            static_cast<double>(statistics.windowLayerFrequency));
        const auto locality = contains(input.recentExperts, candidate.expert.expertId)
            ? 1.0
            : 0.0;
        double cooccurrence{};
        for (const auto& [peer, count] : candidate.cooccurrence) {
            (void)peer;
            cooccurrence = std::max(
                cooccurrence,
                ratio(static_cast<double>(count),
                      static_cast<double>(candidate.frequency)));
        }
        const auto raw = std::max(0.0, 0.50 * transitionProbability +
                                          0.20 * windowPopularity +
                                          0.15 * popularity + 0.05 * locality +
                                          0.10 * cooccurrence);
        const auto evidence = std::min(1.0,
            std::log1p(static_cast<double>(candidate.frequency)) / std::log(9.0));
        const auto agreement = sources.empty()
            ? 0.0
            : static_cast<double>(supportedSources) /
                  static_cast<double>(sources.size());
        const auto confidence = std::clamp(0.65 * evidence + 0.35 * agreement,
                                           0.0, 1.0);
        scores.push_back({candidate.expert.expertId, raw, confidence});
        totalRaw += raw;
    }

    std::vector<ExpertPrediction> predictions;
    predictions.reserve(scores.size());
    for (const auto& score : scores) {
        const auto probability = totalRaw > 0.0
            ? score.raw / totalRaw
            : 1.0 / static_cast<double>(scores.size());
        if (score.confidence >= config_.minimumConfidence) {
            predictions.push_back(
                {score.expertId, probability, score.confidence, targetLayer});
        }
    }
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& left, const auto& right) {
                  if (left.probability != right.probability) {
                      return left.probability > right.probability;
                  }
                  if (left.confidence != right.confidence) {
                      return left.confidence > right.confidence;
                  }
                  return left.expertId < right.expertId;
              });
    if (predictions.size() > config_.maximumPredictions) {
        predictions.resize(config_.maximumPredictions);
    }
    return predictions;
}

std::vector<scheduler::PredictedExpertRequest> ExpertPredictor::predict(
    const scheduler::PredictionInput& input) const {
    const auto detailed = predictDetailed(input);
    std::vector<scheduler::PredictedExpertRequest> result;
    result.reserve(detailed.size());
    for (const auto& prediction : detailed) {
        result.push_back({prediction.expectedLayer, prediction.expertId,
                          prediction.confidence, prediction.probability});
    }
    return result;
}

bool ExpertPredictor::shouldPrefetch(
    const ExpertPrediction& prediction) const noexcept {
    return prediction.confidence >= config_.minimumPrefetchConfidence &&
           prediction.probability > 0.0;
}

std::vector<scheduler::ScheduleHandle> ExpertPredictor::observeAndPrefetch(
    const router::RouterDecision& decision,
    ExpertHistory& history,
    scheduler::Scheduler& scheduler,
    std::uint64_t streamId,
    const std::function<void(const ExpertPrediction&)>& predictionObserver) {
    if (!decision.valid()) throw std::invalid_argument("cannot observe invalid decision");
    {
        std::scoped_lock lock(metricsMutex_);
        const auto pending = pendingByStream_.find(streamId);
        if (pending != pendingByStream_.end() && !pending->second.empty() &&
            pending->second.front().expectedLayer == decision.layerId) {
            ++quality_.opportunities;
            const auto& predicted = pending->second;
            if (contains(decision.selectedExpertIds, predicted.front().expertId)) {
                ++quality_.top1Correct;
            }
            bool covered{};
            for (const auto& item : predicted) {
                if (contains(decision.selectedExpertIds, item.expertId)) {
                    covered = true;
                    ++quality_.usefulPredictions;
                } else {
                    ++quality_.wastedPredictions;
                }
            }
            if (covered) ++quality_.topKCovered;
            pendingByStream_.erase(pending);
        }
    }

    history.record(decision);
    observe(decision, streamId);
    scheduler::PredictionInput input;
    input.currentLayer = decision.layerId;
    input.recentExperts = decision.selectedExpertIds;
    const auto predictions = predictDetailed(input);
    {
        std::scoped_lock lock(metricsMutex_);
        quality_.predictions += predictions.size();
        pendingByStream_[streamId] = predictions;
    }

    std::vector<scheduler::ScheduleHandle> handles;
    handles.reserve(predictions.size());
    for (const auto& prediction : predictions) {
        if (predictionObserver) predictionObserver(prediction);
        if (!shouldPrefetch(prediction)) {
            std::scoped_lock lock(metricsMutex_);
            ++quality_.suppressedPrefetches;
            if (profiler_) profiler_->recordPrefetchSkipped();
            continue;
        }
        try {
            handles.push_back(scheduler.prefetch(
                {prediction.expectedLayer, prediction.expertId,
                 prediction.confidence, prediction.probability}));
        } catch (const std::out_of_range&) {
            // A statistically known expert may be absent from a partial runtime graph.
        }
    }
    return handles;
}

PredictionQualitySnapshot ExpertPredictor::qualitySnapshot() const {
    std::scoped_lock lock(metricsMutex_);
    return quality_;
}

const std::shared_ptr<TransitionDatabase>& ExpertPredictor::database() const noexcept {
    return database_;
}

} // namespace hypermoe::prediction
