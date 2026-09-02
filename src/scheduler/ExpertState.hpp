#pragma once

#include "hypermoe/experts/expert.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hypermoe::scheduler {

enum class ExpertLifecycleState {
    Requested,
    Queued,
    Loading,
    Ready,
    InUse,
    Evicting,
    Failed,
};

[[nodiscard]] std::string_view toString(ExpertLifecycleState state) noexcept;

struct ExpertState {
    LayerId layerId{};
    ExpertId id{};
    MemoryTier currentLocation{MemoryTier::Nvme};
    MemoryTier targetLocation{MemoryTier::Nvme};
    ExpertLifecycleState state{ExpertLifecycleState::Ready};
    std::chrono::steady_clock::time_point lastUsed{};
    std::uint64_t usageCount{};
};

class ExpertResidencyStateMachine {
public:
    void registerExpert(LayerId layerId, ExpertId id,
                        MemoryTier location = MemoryTier::Nvme);
    void registerExpert(ExpertId id, MemoryTier location = MemoryTier::Nvme);
    void request(LayerId layerId, ExpertId id, MemoryTier target);
    void request(ExpertId id, MemoryTier target);
    void markQueued(LayerId layerId, ExpertId id);
    void markQueued(ExpertId id);
    void markLoading(LayerId layerId, ExpertId id);
    void markLoading(ExpertId id);
    void markReady(LayerId layerId, ExpertId id, MemoryTier location);
    void markReady(ExpertId id, MemoryTier location);
    void acquire(LayerId layerId, ExpertId id);
    void acquire(ExpertId id);
    void release(LayerId layerId, ExpertId id);
    void release(ExpertId id);
    void beginEviction(LayerId layerId, ExpertId id, MemoryTier target);
    void beginEviction(ExpertId id, MemoryTier target);
    void markFailed(LayerId layerId, ExpertId id);
    void markFailed(ExpertId id);

    [[nodiscard]] ExpertState snapshot(LayerId layerId, ExpertId id) const;
    [[nodiscard]] ExpertState snapshot(ExpertId id) const;
    [[nodiscard]] std::vector<ExpertState> snapshots() const;

private:
    using StateKey = std::uint64_t;
    [[nodiscard]] static constexpr StateKey key(LayerId layerId,
                                                 ExpertId id) noexcept {
        return (static_cast<StateKey>(layerId) << 32U) | id;
    }
    [[nodiscard]] ExpertState& requireLocked(LayerId layerId, ExpertId id);

    mutable std::mutex mutex_;
    std::unordered_map<StateKey, ExpertState> states_;
};

} // namespace hypermoe::scheduler
