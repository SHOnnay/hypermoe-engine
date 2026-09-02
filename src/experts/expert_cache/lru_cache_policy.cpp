#include "hypermoe/experts/cache_policy.hpp"

#include <limits>

namespace hypermoe {

void LruCachePolicy::onAccess(ExpertId id) {
    lastAccess_[id] = ++clock_;
}

void LruCachePolicy::onResident(ExpertId id, MemoryTier) {
    if (!lastAccess_.contains(id)) {
        lastAccess_.emplace(id, ++clock_);
    }
}

void LruCachePolicy::onEvict(ExpertId, MemoryTier) {}

std::optional<ExpertId>
LruCachePolicy::selectVictim(MemoryTier,
                             std::span<const ExpertId> candidates,
                             const std::unordered_set<ExpertId>& excluded) const {
    std::optional<ExpertId> victim;
    auto oldest = std::numeric_limits<std::uint64_t>::max();
    for (const auto id : candidates) {
        if (excluded.contains(id)) {
            continue;
        }
        const auto it = lastAccess_.find(id);
        const auto stamp = it == lastAccess_.end() ? 0 : it->second;
        if (!victim || stamp < oldest || (stamp == oldest && id < *victim)) {
            victim = id;
            oldest = stamp;
        }
    }
    return victim;
}

} // namespace hypermoe
