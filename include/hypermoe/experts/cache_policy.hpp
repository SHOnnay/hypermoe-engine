#pragma once

#include "hypermoe/experts/expert.hpp"

#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace hypermoe {

class CachePolicy {
public:
    virtual ~CachePolicy() = default;

    virtual void onAccess(ExpertId id) = 0;
    virtual void onResident(ExpertId id, MemoryTier tier) = 0;
    virtual void onEvict(ExpertId id, MemoryTier tier) = 0;
    virtual void setLayerProbability(ExpertId, double) {}
    virtual void setPrefetchConfidence(ExpertId, double) {}

    [[nodiscard]] virtual std::optional<ExpertId>
    selectVictim(MemoryTier tier,
                 std::span<const ExpertId> candidates,
                 const std::unordered_set<ExpertId>& excluded) const = 0;
};

// The Phase 1 baseline remains source-compatible; Phase 2 adds LFU and hybrid
// implementations behind this contract.
class LruCachePolicy final : public CachePolicy {
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
    std::unordered_map<ExpertId, std::uint64_t> lastAccess_;
};

} // namespace hypermoe
