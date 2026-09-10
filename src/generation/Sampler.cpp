#include "generation/Sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hypermoe::generation {

Sampler::Sampler(SamplingConfig config)
    : config_(config), random_(config.seed) {}

std::uint32_t Sampler::sample(std::span<const float> logits) {
    config_.validate(logits.size());
    if (logits.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("sampler vocabulary exceeds token ID range");
    }
    if (config_.strategy == SamplingStrategy::Greedy) {
        for (const auto value : logits) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("sampler requires finite logits");
            }
        }
        return static_cast<std::uint32_t>(
            std::distance(logits.begin(),
                          std::max_element(logits.begin(), logits.end())));
    }
    const auto probabilities = LogitsProcessor::probabilities(logits, config_);
    // Mapping mt19937_64 bits directly keeps seeded sampling identical across
    // standard-library implementations.
    constexpr double denominator = 9007199254740992.0; // 2^53
    const auto draw = static_cast<double>(random_() >> 11U) / denominator;
    double cumulative{};
    std::size_t last{};
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        if (probabilities[index] > 0.0) last = index;
        cumulative += probabilities[index];
        if (draw < cumulative) return static_cast<std::uint32_t>(index);
    }
    return static_cast<std::uint32_t>(last);
}

const SamplingConfig& Sampler::config() const noexcept { return config_; }

} // namespace hypermoe::generation
