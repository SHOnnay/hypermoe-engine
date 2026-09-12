#pragma once

#include "prediction/TransitionDatabase.hpp"
#include "scheduler/Prefetcher.hpp"
#include "scheduler/Scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace hypermoe::prediction {

struct ExpertPrediction {
    ExpertId expertId{};
    double probability{};
    double confidence{};
    LayerId expectedLayer{};
};

struct AdaptivePredictionConfig {
    std::size_t maximumPredictions{3};
    double minimumConfidence{0.05};
    double minimumPrefetchConfidence{0.20};
};

struct PredictionQualitySnapshot {
    std::uint64_t opportunities{};
    std::uint64_t predictions{};
    std::uint64_t top1Correct{};
    std::uint64_t topKCovered{};
    std::uint64_t usefulPredictions{};
    std::uint64_t wastedPredictions{};
    std::uint64_t suppressedPrefetches{};

    [[nodiscard]] double top1Accuracy() const noexcept;
    [[nodiscard]] double topKCoverage() const noexcept;
    [[nodiscard]] double prefetchUsefulness() const noexcept;
};

class ExpertPredictor final : public scheduler::Prefetcher {
public:
    explicit ExpertPredictor(std::shared_ptr<TransitionDatabase> database,
                             std::size_t maximumPredictions = 3,
                             double minimumConfidence = 0.05);
    ExpertPredictor(std::shared_ptr<TransitionDatabase> database,
                    AdaptivePredictionConfig config,
                    std::shared_ptr<Profiler> profiler = {});

    void observe(const router::RouterDecision& decision);
    void observe(const router::RouterDecision& decision, std::uint64_t streamId);
    [[nodiscard]] std::vector<scheduler::PredictedExpertRequest>
    predict(const scheduler::PredictionInput& input) const override;
    [[nodiscard]] std::vector<ExpertPrediction>
    predictDetailed(const scheduler::PredictionInput& input) const;
    [[nodiscard]] bool shouldPrefetch(
        const ExpertPrediction& prediction) const noexcept;
    [[nodiscard]] std::vector<scheduler::ScheduleHandle> observeAndPrefetch(
        const router::RouterDecision& decision,
        ExpertHistory& history,
        scheduler::Scheduler& scheduler,
        std::uint64_t streamId = 0,
        const std::function<void(const ExpertPrediction&)>& predictionObserver = {});
    [[nodiscard]] PredictionQualitySnapshot qualitySnapshot() const;
    [[nodiscard]] const std::shared_ptr<TransitionDatabase>& database() const noexcept;

private:
    std::shared_ptr<TransitionDatabase> database_;
    AdaptivePredictionConfig config_;
    std::shared_ptr<Profiler> profiler_;
    mutable std::mutex metricsMutex_;
    PredictionQualitySnapshot quality_;
    std::unordered_map<std::uint64_t, std::vector<ExpertPrediction>> pendingByStream_;
};

} // namespace hypermoe::prediction
