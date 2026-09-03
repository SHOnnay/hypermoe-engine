#pragma once

#include "hypermoe/experts/expert.hpp"

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

} // namespace hypermoe::router
