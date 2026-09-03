#pragma once

#include "hypermoe/experts/expert.hpp"

#include <cstddef>
#include <vector>

namespace hypermoe {

struct ExpertBatch {
    LayerId layerId{};
    ExpertId expertId{};
    std::vector<std::size_t> tokenIndices;
    std::vector<float> routingWeights;

    void validate(std::size_t tokenCount) const;
    [[nodiscard]] std::size_t size() const noexcept;
};

} // namespace hypermoe
