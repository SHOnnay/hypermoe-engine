#pragma once

#include "hypermoe/experts/cache_policy.hpp"

#include <cstdint>
#include <unordered_map>

namespace hypermoe {

class HybridPolicy final : public CachePolicy {
public:
    void onAccess(ExpertId id) override;
    void onResident(ExpertId id, MemoryTier tier) override;
    void onEvict(ExpertId id, MemoryTier tier) override;
    void setLayerProbability(ExpertId id, double probability) override;
    void setPrefetchConfidence(ExpertId id, double confidence) override;

    [[nodiscard]] std::optional<ExpertId>
    selectVictim(MemoryTier tier,
                 std::span<const ExpertId> candidates,
                 const std::unordered_set<ExpertId>& excluded) const override;

    [[nodiscard]] double score(ExpertId id) const noexcept;

private:
    [[nodiscard]] double normalizedFrequency(ExpertId id) const noexcept;
    [[nodiscard]] double normalizedRecency(ExpertId id) const noexcept;

    std::uint64_t clock_{};
    std::uint64_t maximumFrequency_{};
    std::unordered_map<ExpertId, std::uint64_t> frequency_;
    std::unordered_map<ExpertId, std::uint64_t> lastAccess_;
    std::unordered_map<ExpertId, double> layerProbability_;
    std::unordered_map<ExpertId, double> prefetchConfidence_;
};

} // namespace hypermoe
