#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>

namespace hypermoe {

struct ProfilerSnapshot {
    std::uint64_t tokensProcessed{};
    std::uint64_t expertRequests{};
    std::uint64_t cacheHits{};
    std::uint64_t cacheMisses{};
    std::uint64_t nvmeReads{};
    std::uint64_t nvmeBytes{};
    std::uint64_t ramToVramBytes{};
    std::uint64_t ramEvictions{};
    std::uint64_t vramEvictions{};
    std::uint64_t memoryPressureEvents{};
    std::uint64_t peakVramBytes{};
    std::uint64_t peakRamBytes{};
    std::chrono::nanoseconds transferTime{};
    std::chrono::nanoseconds stallTime{};
    std::chrono::nanoseconds cudaTransferTime{};
    std::chrono::nanoseconds nvmeReadTime{};
    std::chrono::nanoseconds ramCopyTime{};
    std::uint64_t gpuMemoryUsageBytes{};
    std::uint64_t peakGpuMemoryUsageBytes{};
    std::uint64_t transferQueueDepth{};
    std::uint64_t peakTransferQueueDepth{};
    std::uint64_t prefetchRequests{};
    std::uint64_t prefetchHits{};
    std::uint64_t prefetchMisses{};
    std::uint64_t queueWaitSamples{};
    std::chrono::nanoseconds totalQueueWait{};
    std::chrono::nanoseconds overlapEligibleTransferTime{};
    std::chrono::nanoseconds hiddenTransferTime{};
    std::chrono::nanoseconds kernelTime{};
    std::chrono::nanoseconds matmulTime{};
    std::chrono::nanoseconds expertExecutionTime{};
    std::chrono::nanoseconds activationTime{};
    std::chrono::nanoseconds projectionTime{};
    std::chrono::nanoseconds quantizationTime{};
    std::uint64_t tensorAllocations{};
    double gpuUtilizationPercent{};
    double peakGpuUtilizationPercent{};
    double modeledLatencyMs{};

    [[nodiscard]] double cacheHitRate() const noexcept;
    [[nodiscard]] double averageQueueWaitMs() const noexcept;
    [[nodiscard]] double transferOverlapPercentage() const noexcept;
};

class Profiler {
public:
    void recordToken(std::uint64_t count = 1);
    void recordExpertRequest(bool cacheHit);
    void recordNvmeRead(std::uint64_t bytes);
    void recordRamToVram(std::uint64_t bytes);
    void recordRamEvictions(std::uint64_t count = 1);
    void recordVramEvictions(std::uint64_t count = 1);
    void recordMemoryPressureEvent(std::uint64_t count = 1);
    void observeMemory(std::uint64_t vramBytes, std::uint64_t ramBytes);
    void recordTransferTime(std::chrono::nanoseconds duration);
    void recordStallTime(std::chrono::nanoseconds duration);
    void recordModeledLatency(double milliseconds);
    void recordCudaTransferTime(std::chrono::nanoseconds duration);
    void recordNvmeReadTime(std::chrono::nanoseconds duration);
    void recordRamCopyTime(std::chrono::nanoseconds duration);
    void observeGpuMemory(std::uint64_t usedBytes);
    void observeTransferQueueDepth(std::uint64_t depth);
    void recordPrefetchRequest(std::uint64_t count = 1);
    void recordPrefetchHit(std::uint64_t count = 1);
    void recordPrefetchMiss(std::uint64_t count = 1);
    void recordQueueWait(std::chrono::nanoseconds duration);
    void recordTransferOverlap(std::chrono::nanoseconds eligible,
                               std::chrono::nanoseconds hidden);
    void recordKernelTime(std::chrono::nanoseconds duration);
    void recordMatmulTime(std::chrono::nanoseconds duration);
    void recordExpertExecutionTime(std::chrono::nanoseconds duration);
    void recordActivationTime(std::chrono::nanoseconds duration);
    void recordProjectionTime(std::chrono::nanoseconds duration);
    void recordQuantizationTime(std::chrono::nanoseconds duration);
    void recordTensorAllocation(std::uint64_t count = 1);
    void observeGpuUtilization(double percentage);

    [[nodiscard]] ProfilerSnapshot snapshot() const;
    [[nodiscard]] std::string toJson() const;
    void exportJson(const std::filesystem::path& path) const;

private:
    mutable std::mutex mutex_;
    ProfilerSnapshot metrics_;
};

} // namespace hypermoe
