#include "profiling/Profiler.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace hypermoe {

double ProfilerSnapshot::cacheHitRate() const noexcept {
    return expertRequests == 0
               ? 0.0
               : static_cast<double>(cacheHits) / static_cast<double>(expertRequests);
}

double ProfilerSnapshot::averageQueueWaitMs() const noexcept {
    return queueWaitSamples == 0
               ? 0.0
               : std::chrono::duration<double, std::milli>(totalQueueWait).count() /
                     static_cast<double>(queueWaitSamples);
}

double ProfilerSnapshot::transferOverlapPercentage() const noexcept {
    if (overlapEligibleTransferTime.count() <= 0) return 0.0;
    return 100.0 * static_cast<double>(hiddenTransferTime.count()) /
           static_cast<double>(overlapEligibleTransferTime.count());
}

void Profiler::recordToken(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.tokensProcessed += count;
}

void Profiler::recordExpertRequest(bool cacheHit) {
    std::scoped_lock lock(mutex_);
    ++metrics_.expertRequests;
    if (cacheHit) {
        ++metrics_.cacheHits;
    } else {
        ++metrics_.cacheMisses;
    }
}

void Profiler::recordNvmeRead(std::uint64_t bytes) {
    std::scoped_lock lock(mutex_);
    ++metrics_.nvmeReads;
    metrics_.nvmeBytes += bytes;
}

void Profiler::recordRamToVram(std::uint64_t bytes) {
    std::scoped_lock lock(mutex_);
    metrics_.ramToVramBytes += bytes;
}

void Profiler::recordRamEvictions(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.ramEvictions += count;
}

void Profiler::recordVramEvictions(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.vramEvictions += count;
}

void Profiler::recordMemoryPressureEvent(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.memoryPressureEvents += count;
}

void Profiler::observeMemory(std::uint64_t vramBytes, std::uint64_t ramBytes) {
    std::scoped_lock lock(mutex_);
    metrics_.peakVramBytes = std::max(metrics_.peakVramBytes, vramBytes);
    metrics_.peakRamBytes = std::max(metrics_.peakRamBytes, ramBytes);
}

void Profiler::recordTransferTime(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.transferTime += duration;
}

void Profiler::recordStallTime(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.stallTime += duration;
}

void Profiler::recordModeledLatency(double milliseconds) {
    std::scoped_lock lock(mutex_);
    metrics_.modeledLatencyMs += milliseconds;
}

void Profiler::recordCudaTransferTime(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.cudaTransferTime += duration;
}

void Profiler::recordNvmeReadTime(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.nvmeReadTime += duration;
}

void Profiler::recordRamCopyTime(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.ramCopyTime += duration;
}

void Profiler::observeGpuMemory(std::uint64_t usedBytes) {
    std::scoped_lock lock(mutex_);
    metrics_.gpuMemoryUsageBytes = usedBytes;
    metrics_.peakGpuMemoryUsageBytes =
        std::max(metrics_.peakGpuMemoryUsageBytes, usedBytes);
}

void Profiler::observeTransferQueueDepth(std::uint64_t depth) {
    std::scoped_lock lock(mutex_);
    metrics_.transferQueueDepth = depth;
    metrics_.peakTransferQueueDepth =
        std::max(metrics_.peakTransferQueueDepth, depth);
}

void Profiler::recordPrefetchRequest(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.prefetchRequests += count;
}

void Profiler::recordPrefetchHit(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.prefetchHits += count;
}

void Profiler::recordPrefetchMiss(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.prefetchMisses += count;
}

void Profiler::recordQueueWait(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.totalQueueWait += duration;
    ++metrics_.queueWaitSamples;
}

void Profiler::recordTransferOverlap(std::chrono::nanoseconds eligible,
                                     std::chrono::nanoseconds hidden) {
    std::scoped_lock lock(mutex_);
    metrics_.overlapEligibleTransferTime += eligible;
    metrics_.hiddenTransferTime += std::min(eligible, hidden);
}

void Profiler::recordKernelTime(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.kernelTime += duration;
}

void Profiler::recordMatmulTime(std::chrono::nanoseconds duration) {
    std::scoped_lock lock(mutex_);
    metrics_.matmulTime += duration;
}

void Profiler::recordTensorAllocation(std::uint64_t count) {
    std::scoped_lock lock(mutex_);
    metrics_.tensorAllocations += count;
}

void Profiler::observeGpuUtilization(double percentage) {
    const auto bounded = std::clamp(percentage, 0.0, 100.0);
    std::scoped_lock lock(mutex_);
    metrics_.gpuUtilizationPercent = bounded;
    metrics_.peakGpuUtilizationPercent =
        std::max(metrics_.peakGpuUtilizationPercent, bounded);
}

ProfilerSnapshot Profiler::snapshot() const {
    std::scoped_lock lock(mutex_);
    return metrics_;
}

std::string Profiler::toJson() const {
    const auto metrics = snapshot();
    const auto transferMs =
        std::chrono::duration<double, std::milli>(metrics.transferTime).count();
    const auto stallMs =
        std::chrono::duration<double, std::milli>(metrics.stallTime).count();
    const auto cudaTransferMs =
        std::chrono::duration<double, std::milli>(metrics.cudaTransferTime).count();
    const auto nvmeReadMs =
        std::chrono::duration<double, std::milli>(metrics.nvmeReadTime).count();
    const auto ramCopyMs =
        std::chrono::duration<double, std::milli>(metrics.ramCopyTime).count();
    const auto kernelMs =
        std::chrono::duration<double, std::milli>(metrics.kernelTime).count();
    const auto matmulMs =
        std::chrono::duration<double, std::milli>(metrics.matmulTime).count();
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"tokens_processed\": " << metrics.tokensProcessed << ",\n"
           << "  \"expert_requests\": " << metrics.expertRequests << ",\n"
           << "  \"cache_hits\": " << metrics.cacheHits << ",\n"
           << "  \"cache_misses\": " << metrics.cacheMisses << ",\n"
           << "  \"cache_hit_rate\": " << metrics.cacheHitRate() << ",\n"
           << "  \"nvme_reads\": " << metrics.nvmeReads << ",\n"
           << "  \"nvme_bytes\": " << metrics.nvmeBytes << ",\n"
           << "  \"ram_to_vram_bytes\": " << metrics.ramToVramBytes << ",\n"
           << "  \"ram_evictions\": " << metrics.ramEvictions << ",\n"
           << "  \"vram_evictions\": " << metrics.vramEvictions << ",\n"
           << "  \"memory_pressure_events\": " << metrics.memoryPressureEvents << ",\n"
           << "  \"peak_vram_bytes\": " << metrics.peakVramBytes << ",\n"
           << "  \"peak_ram_bytes\": " << metrics.peakRamBytes << ",\n"
           << "  \"transfer_time_ms\": " << transferMs << ",\n"
           << "  \"stall_time_ms\": " << stallMs << ",\n"
           << "  \"cuda_transfer_time_ms\": " << cudaTransferMs << ",\n"
           << "  \"nvme_read_time_ms\": " << nvmeReadMs << ",\n"
           << "  \"ram_copy_time_ms\": " << ramCopyMs << ",\n"
           << "  \"gpu_memory_usage_bytes\": " << metrics.gpuMemoryUsageBytes << ",\n"
           << "  \"peak_gpu_memory_usage_bytes\": "
           << metrics.peakGpuMemoryUsageBytes << ",\n"
           << "  \"transfer_queue_depth\": " << metrics.transferQueueDepth << ",\n"
           << "  \"peak_transfer_queue_depth\": "
           << metrics.peakTransferQueueDepth << ",\n"
           << "  \"prefetch_requests\": " << metrics.prefetchRequests << ",\n"
           << "  \"prefetch_hits\": " << metrics.prefetchHits << ",\n"
           << "  \"prefetch_misses\": " << metrics.prefetchMisses << ",\n"
           << "  \"average_queue_wait_ms\": " << metrics.averageQueueWaitMs() << ",\n"
           << "  \"transfer_overlap_percentage\": "
           << metrics.transferOverlapPercentage() << ",\n"
           << "  \"kernel_time_ms\": " << kernelMs << ",\n"
           << "  \"matmul_time_ms\": " << matmulMs << ",\n"
           << "  \"tensor_allocations\": " << metrics.tensorAllocations << ",\n"
           << "  \"gpu_utilization_percent\": "
           << metrics.gpuUtilizationPercent << ",\n"
           << "  \"peak_gpu_utilization_percent\": "
           << metrics.peakGpuUtilizationPercent << ",\n"
           << "  \"modeled_latency_ms\": " << metrics.modeledLatencyMs << "\n"
           << "}\n";
    return output.str();
}

void Profiler::exportJson(const std::filesystem::path& path) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create profiler report: " + path.string());
    }
    output << toJson();
    if (!output) {
        throw std::runtime_error("failed writing profiler report: " + path.string());
    }
}

} // namespace hypermoe
