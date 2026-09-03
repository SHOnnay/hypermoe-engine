#pragma once

#include "prediction/TransitionDatabase.hpp"
#include "scheduler/Prefetcher.hpp"
#include "scheduler/Scheduler.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace hypermoe::prediction {

class ExpertPredictor final : public scheduler::Prefetcher {
public:
    explicit ExpertPredictor(std::shared_ptr<TransitionDatabase> database,
                             std::size_t maximumPredictions = 3,
                             double minimumConfidence = 0.05);

    void observe(const router::RouterDecision& decision);
    void observe(const router::RouterDecision& decision, std::uint64_t streamId);
    [[nodiscard]] std::vector<scheduler::PredictedExpertRequest>
    predict(const scheduler::PredictionInput& input) const override;
    [[nodiscard]] std::vector<scheduler::ScheduleHandle> observeAndPrefetch(
        const router::RouterDecision& decision,
        ExpertHistory& history,
        scheduler::Scheduler& scheduler);
    [[nodiscard]] const std::shared_ptr<TransitionDatabase>& database() const noexcept;

private:
    std::shared_ptr<TransitionDatabase> database_;
    std::size_t maximumPredictions_;
    double minimumConfidence_;
};

} // namespace hypermoe::prediction
