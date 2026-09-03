#pragma once

#include "hypermoe/experts/expert.hpp"

#include <cstddef>
#include <vector>

namespace hypermoe::router {

struct RouterDecision {
    LayerId layerId{};
    std::vector<ExpertId> selectedExpertIds;
    std::vector<float> routingScores;

    [[nodiscard]] bool valid() const noexcept {
        return !selectedExpertIds.empty() &&
               selectedExpertIds.size() == routingScores.size();
    }
};

struct ExpertTokenGroup {
    ExpertId expertId{};
    std::vector<std::size_t> tokenIndices;
    std::vector<float> routingScores;
};

struct BatchRouterDecision {
    LayerId layerId{};
    std::vector<RouterDecision> tokens;
    std::vector<ExpertTokenGroup> expertGroups;

    [[nodiscard]] bool valid() const noexcept {
        if (tokens.empty()) return false;
        for (const auto& token : tokens) {
            if (!token.valid() || token.layerId != layerId) return false;
        }
        for (const auto& group : expertGroups) {
            if (group.tokenIndices.empty() ||
                group.tokenIndices.size() != group.routingScores.size()) return false;
        }
        return true;
    }
};

} // namespace hypermoe::router
