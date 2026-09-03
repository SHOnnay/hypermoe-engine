#include "experts/ExpertBatch.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace hypermoe {

void ExpertBatch::validate(std::size_t tokenCount) const {
    if (tokenCount == 0 || tokenIndices.empty() ||
        tokenIndices.size() != routingWeights.size()) {
        throw std::invalid_argument("expert batch metadata is incomplete");
    }
    std::unordered_set<std::size_t> seen;
    for (std::size_t index = 0; index < tokenIndices.size(); ++index) {
        if (tokenIndices[index] >= tokenCount ||
            !std::isfinite(routingWeights[index]) ||
            !seen.insert(tokenIndices[index]).second) {
            throw std::invalid_argument("expert batch contains an invalid token assignment");
        }
    }
}

std::size_t ExpertBatch::size() const noexcept { return tokenIndices.size(); }

} // namespace hypermoe
