#pragma once

#include "hypermoe/memory/memory_tier.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace hypermoe {

using AllocationId = std::uint64_t;

struct MemoryAllocation {
    AllocationId id{};
    MemoryTier tier{MemoryTier::Ram};
    std::size_t sizeBytes{};
    std::string tag;
};

struct TierUsage {
    std::size_t usedBytes{};
    std::size_t limitBytes{};

    [[nodiscard]] std::size_t availableBytes() const noexcept {
        return limitBytes - usedBytes;
    }
};

struct MemorySnapshot {
    TierUsage vram;
    TierUsage ram;
};

// Phase 1 allocations are logical reservations. Backends will bind these tokens
// to real CUDA or host buffers without changing the accounting API.
class MemoryManager {
public:
    MemoryManager(std::size_t vramLimitBytes, std::size_t ramLimitBytes);

    [[nodiscard]] std::optional<MemoryAllocation>
    allocate(MemoryTier tier, std::size_t sizeBytes, std::string tag = {});

    [[nodiscard]] bool release(AllocationId id);
    [[nodiscard]] MemorySnapshot snapshot() const;
    [[nodiscard]] TierUsage usage(MemoryTier tier) const;
    [[nodiscard]] bool contains(AllocationId id) const;

private:
    [[nodiscard]] TierUsage& usageFor(MemoryTier tier);
    [[nodiscard]] const TierUsage& usageFor(MemoryTier tier) const;

    mutable std::mutex mutex_;
    TierUsage vram_;
    TierUsage ram_;
    AllocationId nextId_{1};
    std::unordered_map<AllocationId, MemoryAllocation> allocations_;
};

} // namespace hypermoe
