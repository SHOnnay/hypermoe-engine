#include "generation/LogitsProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace hypermoe::generation {

void SamplingConfig::validate(std::size_t vocabularySize) const {
    if ((strategy != SamplingStrategy::Greedy &&
         strategy != SamplingStrategy::Stochastic) ||
        vocabularySize == 0 || !std::isfinite(temperature) || temperature <= 0.0F ||
        (topK != 0 && topK > vocabularySize) || !std::isfinite(topP) ||
        topP <= 0.0F || topP > 1.0F) {
        throw std::invalid_argument("sampling configuration is invalid");
    }
}

std::vector<double> LogitsProcessor::probabilities(
    std::span<const float> logits,
    const SamplingConfig& config) {
    config.validate(logits.size());
    std::vector<std::size_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    for (const auto value : logits) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("sampler requires finite logits");
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                      std::size_t right) {
        return logits[left] > logits[right];
    });
    const auto retained = config.topK == 0 ? logits.size() : config.topK;
    const auto scale = config.strategy == SamplingStrategy::Stochastic
        ? static_cast<double>(config.temperature) : 1.0;
    const auto maximum = static_cast<double>(logits[order.front()]) / scale;
    std::vector<double> result(logits.size(), 0.0);
    double total{};
    for (std::size_t rank = 0; rank < retained; ++rank) {
        const auto index = order[rank];
        result[index] = std::exp(static_cast<double>(logits[index]) / scale - maximum);
        total += result[index];
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::runtime_error("logits softmax normalization failed");
    }
    for (auto& probability : result) probability /= total;

    if (config.topP < 1.0F) {
        double cumulative{};
        std::size_t keep{};
        for (; keep < retained; ++keep) {
            cumulative += result[order[keep]];
            if (cumulative >= static_cast<double>(config.topP)) {
                ++keep;
                break;
            }
        }
        keep = std::max<std::size_t>(1, keep);
        for (std::size_t rank = keep; rank < retained; ++rank) {
            result[order[rank]] = 0.0;
        }
        double keptTotal{};
        for (const auto probability : result) keptTotal += probability;
        for (auto& probability : result) probability /= keptTotal;
    }
    return result;
}

} // namespace hypermoe::generation
