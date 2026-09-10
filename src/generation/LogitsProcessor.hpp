#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace hypermoe::generation {

enum class SamplingStrategy : std::uint8_t {
    Greedy = 0,
    Stochastic = 1,
};

struct SamplingConfig {
    SamplingStrategy strategy{SamplingStrategy::Greedy};
    float temperature{1.0F};
    std::size_t topK{};
    float topP{1.0F};
    std::uint64_t seed{};

    void validate(std::size_t vocabularySize) const;
};

class LogitsProcessor {
public:
    [[nodiscard]] static std::vector<double> probabilities(
        std::span<const float> logits,
        const SamplingConfig& config);
};

} // namespace hypermoe::generation
