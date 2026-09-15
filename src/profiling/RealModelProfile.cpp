#include "profiling/RealModelProfile.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace hypermoe::profiling {
namespace {

double milliseconds(std::chrono::nanoseconds value) {
    return std::chrono::duration<double, std::milli>(value).count();
}

std::string escape(std::string_view value) {
    std::string result;
    for (const auto character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

std::size_t checkedAdd(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error("real model profile memory accounting overflow");
    }
    return left + right;
}

} // namespace

std::string RealModelProfile::toJson() const {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n  \"schema\": \"hypermoe.real-model-profile.v1\",\n"
           << "  \"model\": \"" << escape(modelName) << "\",\n"
           << "  \"architecture\": \"" << escape(sourceArchitecture) << "\",\n"
           << "  \"device\": \"" << tensor::toString(device.type) << "\",\n"
           << "  \"parameter_count\": " << parameterCount << ",\n"
           << "  \"layers\": " << layerCount << ",\n"
           << "  \"experts\": " << expertCount << ",\n"
           << "  \"active_experts\": " << activeExperts << ",\n"
           << "  \"tokens\": " << tokens << ",\n"
           << "  \"model_loading_ms\": " << milliseconds(modelLoadingTime) << ",\n"
           << "  \"first_token_latency_ms\": " << milliseconds(firstTokenLatency) << ",\n"
           << "  \"average_decode_latency_ms\": " << milliseconds(averageDecodeLatency) << ",\n"
           << "  \"total_wall_ms\": " << milliseconds(totalWallTime) << ",\n"
           << "  \"tokens_per_second\": " << tokensPerSecond << ",\n"
           << "  \"vram_usage_bytes\": " << vramUsageBytes << ",\n"
           << "  \"ram_usage_bytes\": " << ramUsageBytes << ",\n"
           << "  \"kv_cache_bytes\": " << kvCacheBytes << ",\n"
           << "  \"static_storage_bytes\": " << staticStorageBytes << ",\n"
           << "  \"static_execution_bytes\": " << staticExecutionBytes << ",\n"
           << "  \"resident_expert_device_bytes\": "
           << residentExpertDeviceBytes << ",\n"
           << "  \"resident_expert_ram_bytes\": "
           << residentExpertRamBytes << ",\n"
           << "  \"expert_requests\": " << expertRequests << ",\n"
           << "  \"cache_hits\": " << cacheHits << ",\n"
           << "  \"cache_misses\": " << cacheMisses << ",\n"
           << "  \"expert_transfer_bytes\": " << expertTransferBytes << ",\n"
           << "  \"cache_hit_rate\": " << cacheHitRate << ",\n"
           << "  \"prefetch_requests\": " << prefetchRequests << ",\n"
           << "  \"prefetch_hits\": " << prefetchHits << ",\n"
           << "  \"prefetch_misses\": " << prefetchMisses << ",\n"
           << "  \"prefetch_accuracy\": " << prefetchAccuracy << ",\n"
           << "  \"expert_device_budget_bytes\": " << expertDeviceBudgetBytes << ",\n"
           << "  \"configured_expert_device_budget_bytes\": " << expertDeviceBudgetPlan.configuredBytes << ",\n"
           << "  \"automatic_expert_device_budget\": " << (expertDeviceBudgetPlan.automatic ? "true" : "false") << ",\n"
           << "  \"expert_device_reserved_bytes\": " << expertDeviceBudgetPlan.reservedBytes << ",\n"
           << "  \"kv_cache_reservation_bytes\": " << expertDeviceBudgetPlan.reservations.kvCacheBytes << ",\n"
           << "  \"cuda_workspace_reservation_bytes\": " << expertDeviceBudgetPlan.reservations.workspaceBytes << ",\n"
           << "  \"cuda_staging_pool_reservation_bytes\": " << expertDeviceBudgetPlan.reservations.stagingPoolBytes << ",\n"
           << "  \"cuda_safety_reservation_bytes\": " << expertDeviceBudgetPlan.reservations.safetyBytes << ",\n"
           << "  \"expert_transfer_allowance_bytes\": " << expertDeviceBudgetPlan.transferAllowanceBytes << ",\n"
           << "  \"expert_ram_budget_bytes\": " << expertRamBudgetBytes << ",\n"
           << "  \"transfer_compute_overlap\": " << (transferComputeOverlap ? "true" : "false") << ",\n"
           << "  \"resident_expert_count\": " << residentExpertCount << ",\n"
           << "  \"device_resident_expert_count\": " << deviceResidentExpertCount << ",\n"
           << "  \"vram_evictions\": " << vramEvictions << ",\n"
           << "  \"vram_promotions\": " << vramPromotions << ",\n"
           << "  \"ram_evictions\": " << ramEvictions << ",\n"
           << "  \"host_to_device_bytes\": " << hostToDeviceBytes << ",\n"
           << "  \"ram_to_vram_transfer_bytes\": " << hostToDeviceBytes << ",\n"
           << "  \"nvme_transfer_bytes\": " << expertTransferBytes << ",\n"
           << "  \"expert_physical_device_bytes\": " << expertPhysicalDeviceBytes << ",\n"
           << "  \"gpu_total_bytes\": " << deviceMemory.totalBytes << ",\n"
           << "  \"gpu_free_bytes\": " << deviceMemory.freeBytes << ",\n"
           << "  \"synchronization_count\": " << synchronizationCount << ",\n"
           << "  \"gpu_utilization_percent\": null,\n"
           << "  \"expert_frequency\": {";
    bool first = true;
    for (const auto& [expert, frequency] : expertFrequency) {
        output << (first ? "\n" : ",\n") << "    \"" << expert << "\": "
               << frequency;
        first = false;
    }
    if (!expertFrequency.empty()) output << '\n' << "  ";
    output << "}\n}\n";
    return output.str();
}

RealModelProfile RealModelProfileCollector::collect(
    const models::ModelManifest& manifest,
    std::uint64_t parameterCount,
    tensor::Device device,
    std::span<const models::runtime::ModelForwardResult> forwards,
    const models::runtime::PackedRuntimeSnapshot& runtime,
    std::size_t kvCacheBytes,
    std::chrono::nanoseconds wallTime) {
    if (forwards.empty() || wallTime <= std::chrono::nanoseconds::zero()) {
        throw std::invalid_argument("real model profile requires measured forwards");
    }
    RealModelProfile result;
    result.modelName = manifest.modelName;
    result.sourceArchitecture = manifest.sourceArchitecture;
    result.device = device;
    result.parameterCount = parameterCount;
    result.layerCount = manifest.config.layerCount;
    result.expertCount = manifest.config.expertCount;
    result.firstTokenLatency = forwards.front().timings.total;
    result.totalWallTime = wallTime;
    result.kvCacheBytes = kvCacheBytes;
    result.staticStorageBytes = runtime.staticStorageBytes;
    result.staticExecutionBytes = runtime.staticExecutionBytes;
    std::chrono::nanoseconds decode{};
    std::set<std::string> active;
    for (std::size_t index = 0; index < forwards.size(); ++index) {
        const auto& forward = forwards[index];
        result.tokens += forward.embeddings.shape().dimensions()[0];
        if (index != 0) decode += forward.timings.total;
        for (const auto& layer : forward.transformer.layers) {
            for (const auto& decision : layer.routing) {
                for (const auto expert : decision.selectedExpertIds) {
                    const auto key = std::to_string(layer.layerId) + ":" +
                                     std::to_string(expert);
                    ++result.expertFrequency[key];
                    active.insert(key);
                }
            }
        }
    }
    result.activeExperts = active.size();
    if (forwards.size() > 1) {
        result.averageDecodeLatency = decode /
            static_cast<std::chrono::nanoseconds::rep>(forwards.size() - 1);
    }
    result.tokensPerSecond = static_cast<double>(result.tokens) /
        std::chrono::duration<double>(wallTime).count();
    result.expertRequests = runtime.profiler.expertRequests;
    result.cacheHits = runtime.profiler.cacheHits;
    result.cacheMisses = runtime.profiler.cacheMisses;
    result.expertTransferBytes = runtime.profiler.nvmeBytes;
    result.cacheHitRate = runtime.profiler.cacheHitRate();
    result.prefetchRequests = runtime.profiler.prefetchRequests;
    result.prefetchHits = runtime.profiler.prefetchHits;
    result.prefetchMisses = runtime.profiler.prefetchMisses;
    result.prefetchAccuracy = result.prefetchRequests == 0 ? 0.0 :
        static_cast<double>(result.prefetchHits) /
        static_cast<double>(result.prefetchRequests);
    const auto expertDevice = runtime.expertMemory.vram.usedBytes;
    const auto expertRam = runtime.expertMemory.ram.usedBytes;
    result.residentExpertDeviceBytes = expertDevice;
    result.residentExpertRamBytes = expertRam;
    result.expertDeviceBudgetBytes = runtime.expertMemory.vram.limitBytes;
    result.expertRamBudgetBytes = runtime.expertMemory.ram.limitBytes;
    result.transferComputeOverlap = runtime.transferComputeOverlap;
    result.expertDeviceBudgetPlan = runtime.expertDeviceBudget;
    if (result.expertDeviceBudgetPlan.configuredBytes == 0) {
        result.expertDeviceBudgetPlan.configuredBytes = result.expertDeviceBudgetBytes;
        result.expertDeviceBudgetPlan.effectiveBytes = result.expertDeviceBudgetBytes;
    }
    if (device.type == tensor::DeviceType::CUDA) {
        result.deviceMemory = runtime.deviceMemory;
        result.expertPhysicalDeviceBytes = runtime.transfers.allocatedBytes;
    }
    result.vramPromotions = runtime.experts.vramPromotions;
    result.vramEvictions = runtime.experts.vramEvictions;
    result.ramEvictions = runtime.experts.ramEvictions;
    result.hostToDeviceBytes = runtime.transfers.hostToDeviceBytes;
    result.synchronizationCount = runtime.transfers.synchronizationCount +
        runtime.tensorBackend.synchronizationCount;
    for (const auto& expert : runtime.residency) {
        if (expert.location != MemoryTier::Nvme) ++result.residentExpertCount;
        if (expert.location == MemoryTier::Vram) ++result.deviceResidentExpertCount;
    }
    if (device.type == tensor::DeviceType::CUDA) {
        result.vramUsageBytes = checkedAdd(
            checkedAdd(runtime.staticExecutionBytes, expertDevice), kvCacheBytes);
        result.ramUsageBytes = expertRam;
    } else {
        result.ramUsageBytes = checkedAdd(
            checkedAdd(runtime.staticExecutionBytes, expertDevice),
            checkedAdd(expertRam, kvCacheBytes));
    }
    return result;
}

} // namespace hypermoe::profiling
