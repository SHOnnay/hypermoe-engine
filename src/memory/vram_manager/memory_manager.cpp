#include "hypermoe/memory/memory_manager.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe {

MemoryManager::MemoryManager(std::size_t vramLimitBytes, std::size_t ramLimitBytes)
    : vram_{0, vramLimitBytes}, ram_{0, ramLimitBytes} {}

std::optional<MemoryAllocation>
MemoryManager::allocate(MemoryTier tier, std::size_t sizeBytes, std::string tag) {
    if (tier == MemoryTier::Nvme) {
        throw std::invalid_argument("NVMe objects are owned by DiskStore, not MemoryManager");
    }
    if (sizeBytes == 0) {
        throw std::invalid_argument("cannot allocate a zero-sized buffer");
    }

    std::scoped_lock lock(mutex_);
    auto& tierUsage = usageFor(tier);
    if (sizeBytes > tierUsage.availableBytes()) {
        return std::nullopt;
    }

    MemoryAllocation allocation{nextId_++, tier, sizeBytes, std::move(tag)};
    allocations_.emplace(allocation.id, allocation);
    tierUsage.usedBytes += sizeBytes;
    return allocation;
}

bool MemoryManager::release(AllocationId id) {
    std::scoped_lock lock(mutex_);
    const auto it = allocations_.find(id);
    if (it == allocations_.end()) {
        return false;
    }
    auto& tierUsage = usageFor(it->second.tier);
    tierUsage.usedBytes -= it->second.sizeBytes;
    allocations_.erase(it);
    return true;
}

MemorySnapshot MemoryManager::snapshot() const {
    std::scoped_lock lock(mutex_);
    return {vram_, ram_};
}

TierUsage MemoryManager::usage(MemoryTier tier) const {
    if (tier == MemoryTier::Nvme) {
        throw std::invalid_argument("NVMe capacity is managed by DiskStore");
    }
    std::scoped_lock lock(mutex_);
    return usageFor(tier);
}

bool MemoryManager::contains(AllocationId id) const {
    std::scoped_lock lock(mutex_);
    return allocations_.contains(id);
}

TierUsage& MemoryManager::usageFor(MemoryTier tier) {
    return tier == MemoryTier::Vram ? vram_ : ram_;
}

const TierUsage& MemoryManager::usageFor(MemoryTier tier) const {
    return tier == MemoryTier::Vram ? vram_ : ram_;
}

} // namespace hypermoe
