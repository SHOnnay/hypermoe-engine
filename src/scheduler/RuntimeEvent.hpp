#pragma once

#include "hypermoe/experts/expert.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace hypermoe::scheduler {

enum class RuntimeEventType {
    ExpertRequested,
    TransferStarted,
    TransferCompleted,
    ExpertReady,
    CacheEvicted,
    TransferFailed,
    CudaTransferStarted,
    CudaTransferCompleted,
    CudaKernelStarted,
    CudaKernelCompleted,
};

[[nodiscard]] std::string_view toString(RuntimeEventType type) noexcept;

struct RuntimeEvent {
    RuntimeEventType type{RuntimeEventType::ExpertRequested};
    ExpertId expertId{};
    LayerId layerId{};
    MemoryTier source{MemoryTier::Nvme};
    MemoryTier destination{MemoryTier::Nvme};
    std::chrono::steady_clock::time_point timestamp{std::chrono::steady_clock::now()};
    std::chrono::nanoseconds duration{};
    std::string message;
};

using EventCallback = std::function<void(const RuntimeEvent&)>;
using SubscriptionId = std::uint64_t;

class RuntimeEventBus {
public:
    [[nodiscard]] SubscriptionId subscribe(EventCallback callback);
    [[nodiscard]] bool unsubscribe(SubscriptionId id);
    void publish(const RuntimeEvent& event) const;

private:
    mutable std::mutex mutex_;
    mutable std::unordered_map<SubscriptionId, EventCallback> subscribers_;
    SubscriptionId nextId_{1};
};

} // namespace hypermoe::scheduler
