#pragma once

#include "hypermoe/experts/expert_manager.hpp"
#include "prediction/ExpertHistory.hpp"
#include "prediction/ExpertPredictor.hpp"
#include "profiling/Profiler.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hypermoe::runtime::metrics {

struct RuntimeMetricsSnapshot {
    std::size_t residentExperts{};
    std::size_t vramResidentExperts{};
    std::size_t ramResidentExperts{};
    std::uint64_t vramUsageBytes{};
    std::uint64_t ramUsageBytes{};
    std::map<std::string, std::uint64_t> expertFrequency;
    double predictionTop1Accuracy{};
    double predictionTopKCoverage{};
    double predictionPrefetchUsefulness{};
    double cacheHitRate{};
    std::uint64_t synchronizationCount{};
    std::uint64_t transferLatencyNanoseconds{};
    std::uint64_t kernelLatencyNanoseconds{};
    std::uint64_t usefulPrefetches{};
    std::uint64_t wastedPrefetches{};

    [[nodiscard]] std::string toJson() const;
};

[[nodiscard]] RuntimeMetricsSnapshot collectRuntimeMetrics(
    const MemorySnapshot& memory,
    const prediction::ExpertHistorySnapshot& history,
    const prediction::PredictionQualitySnapshot& predictions,
    const ProfilerSnapshot& profiler,
    const std::vector<ExpertResidencyInfo>& residency);

} // namespace hypermoe::runtime::metrics
