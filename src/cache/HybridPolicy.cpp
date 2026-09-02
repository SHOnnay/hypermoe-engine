#include "cache/HybridPolicy.hpp"

#include <algorithm>
#include <limits>

namespace hypermoe {
namespace {

double lookupOrZero(const std::unordered_map<ExpertId, double>& values,
                    ExpertId id) noexcept {
    const auto it = values.find(id);
    return it == values.end() ? 0.0 : it->second;
}

} // namespace

void HybridPolicy::onAccess(ExpertId id) {
    const auto value = ++frequency_[id];
    maximumFrequency_ = std::max(maximumFrequency_, value);
    lastAccess_[id] = ++clock_;
}

void HybridPolicy::onResident(ExpertId id, MemoryTier) {
    frequency_.try_emplace(id, 0);
    if (!lastAccess_.contains(id)) {
        lastAccess_.emplace(id, ++clock_);
    }
}

void HybridPolicy::onEvict(ExpertId, MemoryTier) {}

void HybridPolicy::setLayerProbability(ExpertId id, double probability) {
    layerProbability_[id] = std::clamp(probability, 0.0, 1.0);
}

void HybridPolicy::setPrefetchConfidence(ExpertId id, double confidence) {
    prefetchConfidence_[id] = std::clamp(confidence, 0.0, 1.0);
}

std::optional<ExpertId>
HybridPolicy::selectVictim(MemoryTier,
                           std::span<const ExpertId> candidates,
                           const std::unordered_set<ExpertId>& excluded) const {
    std::optional<ExpertId> victim;
    auto minimumScore = std::numeric_limits<double>::infinity();
    for (const auto id : candidates) {
        if (excluded.contains(id)) {
            continue;
        }
        const auto candidateScore = score(id);
        if (!victim || candidateScore < minimumScore ||
            (candidateScore == minimumScore && id < *victim)) {
            victim = id;
            minimumScore = candidateScore;
        }
    }
    return victim;
}

double HybridPolicy::score(ExpertId id) const noexcept {
    return 0.4 * normalizedFrequency(id) + 0.3 * normalizedRecency(id) +
           0.2 * lookupOrZero(layerProbability_, id) +
           0.1 * lookupOrZero(prefetchConfidence_, id);
}

double HybridPolicy::normalizedFrequency(ExpertId id) const noexcept {
    if (maximumFrequency_ == 0) {
        return 0.0;
    }
    const auto it = frequency_.find(id);
    return it == frequency_.end()
               ? 0.0
               : static_cast<double>(it->second) /
                     static_cast<double>(maximumFrequency_);
}

double HybridPolicy::normalizedRecency(ExpertId id) const noexcept {
    if (clock_ == 0) {
        return 0.0;
    }
    const auto it = lastAccess_.find(id);
    return it == lastAccess_.end()
               ? 0.0
               : static_cast<double>(it->second) / static_cast<double>(clock_);
}

} // namespace hypermoe
