#include "runtime/metrics/RuntimeMetrics.hpp"

#include <iomanip>
#include <sstream>

namespace hypermoe::runtime::metrics {

RuntimeMetricsSnapshot collectRuntimeMetrics(
    const MemorySnapshot& memory,
    const prediction::ExpertHistorySnapshot& history,
    const prediction::PredictionQualitySnapshot& predictions,
    const ProfilerSnapshot& profiler,
    const std::vector<ExpertResidencyInfo>& residency) {
    RuntimeMetricsSnapshot result;
    result.vramUsageBytes = memory.vram.usedBytes;
    result.ramUsageBytes = memory.ram.usedBytes;
    for (const auto& expert : residency) {
        if (expert.location == MemoryTier::Vram) {
            ++result.vramResidentExperts;
            ++result.residentExperts;
        } else if (expert.location == MemoryTier::Ram) {
            ++result.ramResidentExperts;
            ++result.residentExperts;
        }
    }
    for (const auto& [selection, frequency] : history.usageFrequency) {
        result.expertFrequency.emplace(
            std::to_string(selection.layerId) + ":" +
                std::to_string(selection.expertId),
            frequency);
    }
    result.predictionTop1Accuracy = predictions.top1Accuracy();
    result.predictionTopKCoverage = predictions.topKCoverage();
    result.predictionPrefetchUsefulness = predictions.prefetchUsefulness();
    result.cacheHitRate = profiler.cacheHitRate();
    result.synchronizationCount = profiler.synchronizationCount;
    result.transferLatencyNanoseconds =
        static_cast<std::uint64_t>(profiler.transferTime.count());
    result.kernelLatencyNanoseconds =
        static_cast<std::uint64_t>(profiler.kernelTime.count());
    result.usefulPrefetches = profiler.prefetchUseful;
    result.wastedPrefetches = profiler.prefetchWasted;
    return result;
}

std::string RuntimeMetricsSnapshot::toJson() const {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n  \"schema\": \"hypermoe.runtime-metrics.v1\",\n"
           << "  \"resident_experts\": " << residentExperts << ",\n"
           << "  \"vram_resident_experts\": " << vramResidentExperts << ",\n"
           << "  \"ram_resident_experts\": " << ramResidentExperts << ",\n"
           << "  \"vram_usage_bytes\": " << vramUsageBytes << ",\n"
           << "  \"ram_usage_bytes\": " << ramUsageBytes << ",\n"
           << "  \"prediction_top1_accuracy\": "
           << predictionTop1Accuracy << ",\n"
           << "  \"prediction_topk_coverage\": "
           << predictionTopKCoverage << ",\n"
           << "  \"prediction_prefetch_usefulness\": "
           << predictionPrefetchUsefulness << ",\n"
           << "  \"cache_hit_rate\": " << cacheHitRate << ",\n"
           << "  \"synchronization_count\": " << synchronizationCount << ",\n"
           << "  \"transfer_latency_ns\": " << transferLatencyNanoseconds << ",\n"
           << "  \"kernel_latency_ns\": " << kernelLatencyNanoseconds << ",\n"
           << "  \"useful_prefetches\": " << usefulPrefetches << ",\n"
           << "  \"wasted_prefetches\": " << wastedPrefetches << ",\n"
           << "  \"expert_frequency\": {";
    bool first = true;
    for (const auto& [expert, frequency] : expertFrequency) {
        output << (first ? "\n" : ",\n") << "    \"" << expert << "\": "
               << frequency;
        first = false;
    }
    if (!expertFrequency.empty()) output << "\n  ";
    output << "}\n}\n";
    return output.str();
}

} // namespace hypermoe::runtime::metrics
