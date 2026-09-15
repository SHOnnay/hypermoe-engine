#pragma once

#include "backend/Backend.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "models/ModelManifest.hpp"
#include "models/runtime/ModelRuntime.hpp"
#include "models/runtime/ExpertDeviceBudget.hpp"
#include "prediction/ExpertHistory.hpp"
#include "profiling/Profiler.hpp"
#include "runtime/cache/KVCache.hpp"
#include "runtime/metrics/RuntimeMetrics.hpp"
#include "tensor/Tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

namespace hypermoe::models::runtime {

struct PackedRuntimeConfiguration {
    tensor::Device device{tensor::Device::cpu()};
    std::size_t expertDeviceBudgetBytes{512U * 1024U * 1024U};
    std::size_t expertRamBudgetBytes{2U * 1024U * 1024U * 1024U};
    std::size_t transferWorkers{2};
    std::size_t schedulerWorkers{2};
    bool adaptivePrediction{true};
    bool adaptiveResidency{true};
    double minimumPrefetchConfidence{0.20};
    bool transferComputeOverlap{true};
    bool automaticExpertDeviceBudget{false};
    ExpertDeviceReservations expertDeviceReservations;

    void validate() const;
    [[nodiscard]] static std::size_t parseBudgetBytes(std::string_view text);
};

struct PackedRuntimeSnapshot {
    MemorySnapshot expertMemory;
    ExpertManagerStats experts;
    ProfilerSnapshot profiler;
    prediction::ExpertHistorySnapshot history;
    prediction::PredictionQualitySnapshot prediction;
    std::vector<ExpertResidencyInfo> residency;
    backend::BackendStats transfers;
    std::size_t staticStorageBytes{};
    std::size_t staticExecutionBytes{};
    backend::BackendStats tensorBackend;
    bool transferComputeOverlap{true};
    ExpertDeviceBudgetPlan expertDeviceBudget;
    backend::MemoryInfo deviceMemory;
};

class PackedModelRuntime {
public:
    explicit PackedModelRuntime(
        const std::filesystem::path& artifact,
        PackedRuntimeConfiguration configuration = {});
    ~PackedModelRuntime();

    PackedModelRuntime(const PackedModelRuntime&) = delete;
    PackedModelRuntime& operator=(const PackedModelRuntime&) = delete;

    [[nodiscard]] ModelForwardResult forward(
        std::span<const std::uint32_t> tokenIds,
        std::uint64_t sequencePosition = 0);
    [[nodiscard]] ModelForwardResult forward(
        std::span<const std::uint32_t> tokenIds,
        std::uint64_t sequencePosition,
        hypermoe::runtime::cache::KVCacheBase& cache);
    [[nodiscard]] std::shared_ptr<hypermoe::runtime::cache::KVCacheBase>
    createKVCache(std::size_t maximumSequenceLength) const;

    [[nodiscard]] const models::ModelManifest& manifest() const noexcept;
    [[nodiscard]] const ModelArchitecture& architecture() const noexcept;
    [[nodiscard]] tensor::Device device() const noexcept;
    [[nodiscard]] tensor::Tensor materializeHost(tensor::TensorView value) const;
    [[nodiscard]] PackedRuntimeSnapshot snapshot() const;
    [[nodiscard]] hypermoe::runtime::metrics::RuntimeMetricsSnapshot metrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hypermoe::models::runtime
