#include "cache/LFUPolicy.hpp"

#include <limits>

namespace hypermoe {

void LFUPolicy::onAccess(ExpertId id) {
    ++frequency_[id];
    lastAccess_[id] = ++clock_;
}

void LFUPolicy::onResident(ExpertId id, MemoryTier) {
    frequency_.try_emplace(id, 0);
    if (!lastAccess_.contains(id)) {
        lastAccess_.emplace(id, ++clock_);
    }
}

void LFUPolicy::onEvict(ExpertId, MemoryTier) {}

std::optional<ExpertId>
LFUPolicy::selectVictim(MemoryTier,
                        std::span<const ExpertId> candidates,
                        const std::unordered_set<ExpertId>& excluded) const {
    std::optional<ExpertId> victim;
    auto minimumFrequency = std::numeric_limits<std::uint64_t>::max();
    auto oldest = std::numeric_limits<std::uint64_t>::max();
    for (const auto id : candidates) {
        if (excluded.contains(id)) {
            continue;
        }
        const auto frequencyIt = frequency_.find(id);
        const auto recencyIt = lastAccess_.find(id);
        const auto frequency = frequencyIt == frequency_.end() ? 0 : frequencyIt->second;
        const auto recency = recencyIt == lastAccess_.end() ? 0 : recencyIt->second;
        if (!victim || frequency < minimumFrequency ||
            (frequency == minimumFrequency && recency < oldest) ||
            (frequency == minimumFrequency && recency == oldest && id < *victim)) {
            victim = id;
            minimumFrequency = frequency;
            oldest = recency;
        }
    }
    return victim;
}

} // namespace hypermoe
