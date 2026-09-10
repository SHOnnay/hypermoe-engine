#pragma once

#include "generation/LogitsProcessor.hpp"

#include <cstdint>
#include <random>
#include <span>

namespace hypermoe::generation {

class Sampler {
public:
    explicit Sampler(SamplingConfig config = {});
    [[nodiscard]] std::uint32_t sample(std::span<const float> logits);
    [[nodiscard]] const SamplingConfig& config() const noexcept;

private:
    SamplingConfig config_;
    std::mt19937_64 random_;
};

} // namespace hypermoe::generation
