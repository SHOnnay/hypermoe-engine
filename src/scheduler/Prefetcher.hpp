#pragma once

#include "hypermoe/experts/expert.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hypermoe::scheduler {

using WorkloadPattern = std::unordered_map<LayerId, std::vector<ExpertId>>;

struct PredictionInput {
    LayerId currentLayer{};
    std::vector<ExpertId> recentExperts;
    WorkloadPattern workloadPattern;
};

struct PredictedExpertRequest {
    LayerId layerId{};
    ExpertId expertId{};
    double confidence{};
};

class Prefetcher {
public:
    virtual ~Prefetcher() = default;
    [[nodiscard]] virtual std::vector<PredictedExpertRequest>
    predict(const PredictionInput& input) const = 0;
};

class LocalityPrefetcher final : public Prefetcher {
public:
    explicit LocalityPrefetcher(std::size_t maximumPredictions = 3);

    [[nodiscard]] std::vector<PredictedExpertRequest>
    predict(const PredictionInput& input) const override;

private:
    std::size_t maximumPredictions_;
};

} // namespace hypermoe::scheduler
