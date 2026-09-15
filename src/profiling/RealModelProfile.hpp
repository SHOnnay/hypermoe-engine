#pragma once

#include "models/ModelManifest.hpp"
#include "models/runtime/ModelRuntime.hpp"
#include "models/runtime/PackedModelRuntime.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>

namespace hypermoe::profiling {

struct RealModelProfile {
    std::string modelName;
    std::string sourceArchitecture;
    tensor::Device device;
    std::uint64_t parameterCount{};
    std::size_t layerCount{};
    std::size_t expertCount{};
    std::size_t activeExperts{};
    std::size_t tokens{};
    std::chrono::nanoseconds modelLoadingTime{};
    std::chrono::nanoseconds firstTokenLatency{};
    std::chrono::nanoseconds averageDecodeLatency{};
    std::chrono::nanoseconds totalWallTime{};
    double tokensPerSecond{};
    std::size_t vramUsageBytes{};
    std::size_t ramUsageBytes{};
    std::size_t kvCacheBytes{};
    std::size_t staticStorageBytes{};
    std::size_t staticExecutionBytes{};
    std::size_t residentExpertDeviceBytes{};
    std::size_t residentExpertRamBytes{};
    std::uint64_t expertRequests{};
    std::uint64_t cacheHits{};
    std::uint64_t cacheMisses{};
    std::uint64_t expertTransferBytes{};
    double cacheHitRate{};
    std::uint64_t prefetchRequests{};
    std::uint64_t prefetchHits{};
    std::uint64_t prefetchMisses{};
    double prefetchAccuracy{};
    std::size_t expertDeviceBudgetBytes{};
    std::size_t expertRamBudgetBytes{};
    std::size_t residentExpertCount{};
    std::size_t deviceResidentExpertCount{};
    std::uint64_t vramEvictions{};
    std::uint64_t ramEvictions{};
    std::uint64_t hostToDeviceBytes{};
    std::uint64_t synchronizationCount{};
    bool transferComputeOverlap{};
    models::runtime::ExpertDeviceBudgetPlan expertDeviceBudgetPlan;
    backend::MemoryInfo deviceMemory;
    std::uint64_t vramPromotions{};
    std::uint64_t expertPhysicalDeviceBytes{};
    std::map<std::string, std::uint64_t> expertFrequency;

    [[nodiscard]] std::string toJson() const;
};

class RealModelProfileCollector {
public:
    [[nodiscard]] static RealModelProfile collect(
        const models::ModelManifest& manifest,
        std::uint64_t parameterCount,
        tensor::Device device,
        std::span<const models::runtime::ModelForwardResult> forwards,
        const models::runtime::PackedRuntimeSnapshot& runtime,
        std::size_t kvCacheBytes,
        std::chrono::nanoseconds wallTime);
};

} // namespace hypermoe::profiling
