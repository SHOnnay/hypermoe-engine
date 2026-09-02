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
    double modeledLatencyMs{};

    [[nodiscard]] double cacheHitRate() const noexcept;
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

    [[nodiscard]] ProfilerSnapshot snapshot() const;
    [[nodiscard]] std::string toJson() const;
    void exportJson(const std::filesystem::path& path) const;

private:
    mutable std::mutex mutex_;
    ProfilerSnapshot metrics_;
};

} // namespace hypermoe
