#pragma once

#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "scheduler/ExpertState.hpp"
#include "scheduler/Prefetcher.hpp"
#include "scheduler/RuntimeEvent.hpp"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hypermoe::scheduler {

enum class TransferPriority : int {
    BackgroundMaintenance = 100,
    CacheWarming = 200,
    PredictedNextLayer = 300,
    ActiveInference = 400,
};

struct ScheduleRequest {
    LayerId layerId{};
    ExpertId expertId{};
    MemoryTier source{MemoryTier::Nvme};
    MemoryTier destination{MemoryTier::Vram};
    TransferPriority priority{TransferPriority::ActiveInference};
    std::shared_ptr<const std::vector<std::byte>> hostBuffer;
    std::shared_ptr<PinnedBuffer> pinnedBuffer;
    std::shared_ptr<backend::DeviceBuffer> deviceBuffer;
    bool eviction{};
};

struct ScheduleResult {
    bool success{};
    bool cancelled{};
    TransferResult transfer;
    ExpertState state;
    std::string error;
};

using ScheduleCallback = std::function<void(const ScheduleResult&)>;

class ScheduleHandle {
public:
    ScheduleHandle() = default;
    ScheduleHandle(std::shared_future<ScheduleResult> future,
                   std::shared_ptr<std::atomic_bool> cancelled);

    void cancel() noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const std::shared_future<ScheduleResult>& future() const noexcept;

private:
    std::shared_future<ScheduleResult> future_;
    std::shared_ptr<std::atomic_bool> cancelled_;
};

class Scheduler {
public:
    Scheduler(std::shared_ptr<TransferManager> transfers,
              std::shared_ptr<Profiler> profiler = {},
              std::size_t workerCount = 2);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void registerExpert(LayerId layerId, ExpertId id,
                        MemoryTier location = MemoryTier::Nvme);
    void registerExpert(ExpertId id, MemoryTier location = MemoryTier::Nvme);
    [[nodiscard]] ScheduleHandle schedule(ScheduleRequest request,
                                          ScheduleCallback callback = {});
    [[nodiscard]] ScheduleHandle prefetch(const PredictedExpertRequest& prediction,
                                          ScheduleCallback callback = {});
    [[nodiscard]] std::vector<ScheduleHandle>
    prefetch(const Prefetcher& prefetcher,
             const PredictionInput& input,
             ScheduleCallback callback = {});

    void acquire(LayerId layerId, ExpertId id);
    void acquire(ExpertId id);
    void release(LayerId layerId, ExpertId id);
    void release(ExpertId id);
    [[nodiscard]] ExpertState state(LayerId layerId, ExpertId id) const;
    [[nodiscard]] ExpertState state(ExpertId id) const;
    [[nodiscard]] std::size_t pending() const;
    [[nodiscard]] RuntimeEventBus& events() noexcept;
    [[nodiscard]] const ExpertResidencyStateMachine& states() const noexcept;
    void shutdown();

private:
    using ExpertKey = std::uint64_t;
    [[nodiscard]] static constexpr ExpertKey key(LayerId layerId,
                                                  ExpertId id) noexcept {
        return (static_cast<ExpertKey>(layerId) << 32U) | id;
    }

    struct Task {
        ScheduleRequest request;
        std::vector<ScheduleCallback> callbacks;
        std::promise<ScheduleResult> promise;
        std::shared_future<ScheduleResult> future;
        std::shared_ptr<std::atomic_bool> cancelled;
        std::chrono::steady_clock::time_point enqueuedAt;
        std::uint64_t generation{};
        bool started{};
        bool completed{};
        bool prefetchRequest{};
        bool activeConsumer{};
    };

    struct QueueEntry {
        std::shared_ptr<Task> task;
        TransferPriority priority{TransferPriority::BackgroundMaintenance};
        std::uint64_t sequence{};
        std::uint64_t generation{};
    };

    struct HigherPriority {
        bool operator()(const QueueEntry& left, const QueueEntry& right) const noexcept;
    };

    [[nodiscard]] ScheduleHandle readyResult(const ScheduleRequest& request,
                                             ScheduleCallback callback);
    void workerLoop();
    void complete(const std::shared_ptr<Task>& task, ScheduleResult result);
    void publish(RuntimeEventType type,
                 const ScheduleRequest& request,
                 std::chrono::nanoseconds duration = {},
                 std::string message = {});

    std::shared_ptr<TransferManager> transfers_;
    std::shared_ptr<Profiler> profiler_;
    ExpertResidencyStateMachine states_;
    RuntimeEventBus events_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, HigherPriority> queue_;
    std::unordered_map<ExpertKey, std::shared_ptr<Task>> pendingByExpert_;
    std::unordered_map<ExpertKey, TransferResult> residentTransfers_;
    std::unordered_set<ExpertKey> prefetchedReady_;
    std::vector<std::thread> workers_;
    std::uint64_t nextSequence_{};
    bool stopping_{};
};

} // namespace hypermoe::scheduler
