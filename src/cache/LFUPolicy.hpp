#pragma once

#include "hypermoe/experts/cache_policy.hpp"

#include <cstdint>
#include <unordered_map>

namespace hypermoe {

class LFUPolicy final : public CachePolicy {
public:
    void onAccess(ExpertId id) override;
    void onResident(ExpertId id, MemoryTier tier) override;
    void onEvict(ExpertId id, MemoryTier tier) override;

    [[nodiscard]] std::optional<ExpertId>
    selectVictim(MemoryTier tier,
                 std::span<const ExpertId> candidates,
                 const std::unordered_set<ExpertId>& excluded) const override;

private:
    std::uint64_t clock_{};
    std::unordered_map<ExpertId, std::uint64_t> frequency_;
    std::unordered_map<ExpertId, std::uint64_t> lastAccess_;
};

} // namespace hypermoe
