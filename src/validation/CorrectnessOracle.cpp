#include "validation/CorrectnessOracle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace hypermoe::validation {

NumericalTolerance CorrectnessOracle::toleranceFor(tensor::DType dtype) noexcept {
    switch (dtype) {
    case tensor::DType::FP32: return {1.0e-5F, 1.0e-5F};
    case tensor::DType::FP16: return {5.0e-3F, 5.0e-3F};
    case tensor::DType::BF16: return {1.0e-2F, 1.0e-2F};
    case tensor::DType::INT8: return {2.0e-2F, 2.0e-2F};
    }
    return {0.0F, 0.0F};
}

ComparisonResult CorrectnessOracle::compare(
    std::span<const float> actual,
    std::span<const float> expected,
    NumericalTolerance tolerance) {
    if (actual.size() != expected.size() || tolerance.absolute < 0.0F ||
        tolerance.relative < 0.0F) {
        throw std::invalid_argument("oracle comparison metadata is invalid");
    }
    ComparisonResult result{true, 0, 0.0F, 0.0F};
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto absolute = std::abs(actual[index] - expected[index]);
        const auto scale = std::max(std::abs(expected[index]),
                                    std::numeric_limits<float>::min());
        const auto relative = absolute / scale;
        result.maximumAbsoluteError = std::max(result.maximumAbsoluteError, absolute);
        result.maximumRelativeError = std::max(result.maximumRelativeError, relative);
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
            absolute > tolerance.absolute + tolerance.relative * std::abs(expected[index])) {
            result.matches = false;
            ++result.mismatchCount;
        }
    }
    return result;
}

router::RouterDecision CorrectnessOracle::route(
    LayerId layerId,
    std::span<const float> hidden,
    std::span<const float> weights,
    const router::RouterConfig& config) {
    config.validate();
    auto scores = routerLogits(hidden, weights, config.expertCount);
    if (config.normalization == router::RoutingNormalization::Softmax) {
        const auto maximum = *std::max_element(scores.begin(), scores.end());
        double sum{};
        for (auto& score : scores) {
            score = std::exp(score - maximum);
            sum += score;
        }
        if (!std::isfinite(sum) || sum <= 0.0) {
            throw std::runtime_error("oracle router softmax failed");
        }
        for (auto& score : scores) score = static_cast<float>(score / sum);
    }
    std::vector<ExpertId> order(config.expertCount);
    std::iota(order.begin(), order.end(), ExpertId{});
    std::stable_sort(order.begin(), order.end(), [&](ExpertId left, ExpertId right) {
        return scores[left] == scores[right] ? left < right : scores[left] > scores[right];
    });
    order.resize(config.topK);
    router::RouterDecision decision;
    decision.layerId = layerId;
    decision.selectedExpertIds = order;
    for (const auto expert : order) decision.routingScores.push_back(scores[expert]);
    if (config.renormalizeSelected) {
        const auto sum = std::accumulate(decision.routingScores.begin(),
                                         decision.routingScores.end(), 0.0F);
        if (!std::isfinite(sum) || sum == 0.0F) {
            throw std::runtime_error("oracle selected scores cannot be renormalized");
        }
        for (auto& score : decision.routingScores) score /= sum;
    }
    return decision;
}

std::vector<float> CorrectnessOracle::routerLogits(
    std::span<const float> hidden,
    std::span<const float> weights,
    std::size_t expertCount) {
    if (hidden.empty() || expertCount == 0 ||
        expertCount > std::numeric_limits<std::size_t>::max() / hidden.size() ||
        weights.size() != hidden.size() * expertCount) {
        throw std::invalid_argument("oracle router matrix has invalid dimensions");
    }
    std::vector<float> scores(expertCount);
    for (std::size_t expert = 0; expert < expertCount; ++expert) {
        double score{};
        for (std::size_t feature = 0; feature < hidden.size(); ++feature) {
            score += static_cast<double>(hidden[feature]) *
                     weights[feature * expertCount + expert];
        }
        scores[expert] = static_cast<float>(score);
        if (!std::isfinite(scores[expert])) {
            throw std::runtime_error("oracle router produced a non-finite score");
        }
    }
    return scores;
}

std::vector<float> CorrectnessOracle::expertMlp(
    std::span<const float> input,
    std::size_t tokens,
    std::size_t hiddenSize,
    std::span<const float> gate,
    std::span<const float> up,
    std::span<const float> down,
    std::size_t intermediateSize) {
    return expertMlpTrace(input, tokens, hiddenSize, gate, up, down,
                          intermediateSize).output;
}

ExpertOracleTrace CorrectnessOracle::expertMlpTrace(
    std::span<const float> input,
    std::size_t tokens,
    std::size_t hiddenSize,
    std::span<const float> gate,
    std::span<const float> up,
    std::span<const float> down,
    std::size_t intermediateSize) {
    const auto product = [](std::size_t left, std::size_t right) {
        if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
            throw std::invalid_argument("oracle expert dimensions overflow");
        }
        return left * right;
    };
    if (tokens == 0 || hiddenSize == 0 || intermediateSize == 0 ||
        input.size() != product(tokens, hiddenSize) ||
        gate.size() != product(hiddenSize, intermediateSize) ||
        up.size() != product(hiddenSize, intermediateSize) ||
        down.size() != product(intermediateSize, hiddenSize)) {
        throw std::invalid_argument("oracle expert matrices have invalid dimensions");
    }
    ExpertOracleTrace trace;
    trace.gateProjection.resize(tokens * intermediateSize);
    trace.upProjection.resize(tokens * intermediateSize);
    trace.activation.resize(tokens * intermediateSize);
    trace.gated.resize(tokens * intermediateSize);
    trace.output.assign(tokens * hiddenSize, 0.0F);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t intermediate = 0; intermediate < intermediateSize; ++intermediate) {
            double gateValue{};
            double upValue{};
            for (std::size_t hidden = 0; hidden < hiddenSize; ++hidden) {
                const auto inputValue = input[token * hiddenSize + hidden];
                gateValue += inputValue * gate[hidden * intermediateSize + intermediate];
                upValue += inputValue * up[hidden * intermediateSize + intermediate];
            }
            const auto gateFloat = static_cast<float>(gateValue);
            const auto silu = gateFloat / (1.0F + std::exp(-gateFloat));
            const auto position = token * intermediateSize + intermediate;
            trace.gateProjection[position] = gateFloat;
            trace.upProjection[position] = static_cast<float>(upValue);
            trace.activation[position] = silu;
            trace.gated[position] = silu * static_cast<float>(upValue);
        }
        for (std::size_t hidden = 0; hidden < hiddenSize; ++hidden) {
            double value{};
            for (std::size_t intermediate = 0; intermediate < intermediateSize; ++intermediate) {
                value += trace.gated[token * intermediateSize + intermediate] *
                         down[intermediate * hiddenSize + hidden];
            }
            trace.output[token * hiddenSize + hidden] = static_cast<float>(value);
        }
    }
    return trace;
}

} // namespace hypermoe::validation
