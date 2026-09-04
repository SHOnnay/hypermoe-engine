#include "validation/CorrectnessOracle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace hypermoe::validation {

bool CpuCudaComparisonReport::matches() const noexcept {
    return expertSelectionMatches && routerLogits.matches && routingScores.matches &&
           gateProjection.matches && upProjection.matches && activation.matches &&
           gated.matches && finalOutput.matches;
}

bool ExpertCombinationComparisonReport::matches() const noexcept {
    if (!routingMatches || !normalizedWeights || !combinedOutput.matches) return false;
    return std::all_of(individualOutputs.begin(), individualOutputs.end(),
                       [](const auto& result) { return result.matches; });
}

bool ModelLayerComparisonReport::matches() const noexcept {
    return !layers.empty() &&
           std::all_of(layers.begin(), layers.end(),
                       [](const auto& result) { return result.matches; });
}

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
    return selectFromLogits(layerId,
                            routerLogits(hidden, weights, config.expertCount),
                            config);
}

router::RouterDecision CorrectnessOracle::selectFromLogits(
    LayerId layerId,
    std::span<const float> logits,
    const router::RouterConfig& config) {
    config.validate();
    if (logits.size() != config.expertCount) {
        throw std::invalid_argument("oracle logits do not match router expert count");
    }
    std::vector<float> scores(logits.begin(), logits.end());
    if (std::any_of(scores.begin(), scores.end(), [](float score) {
            return !std::isfinite(score);
        })) {
        throw std::invalid_argument("oracle logits must be finite");
    }
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

CpuCudaComparisonReport CorrectnessOracle::compareCpuCuda(
    std::span<const float> cpuRouterLogits,
    std::span<const float> cudaRouterLogits,
    const router::RouterDecision& cpuRouting,
    const router::RouterDecision& cudaRouting,
    const ExpertOracleTrace& cpuTrace,
    const ExpertOracleTrace& cudaTrace,
    tensor::DType executionDType) {
    if (cpuRouting.layerId != cudaRouting.layerId) {
        throw std::invalid_argument("CPU/CUDA routing layers do not match");
    }
    const auto tolerance = toleranceFor(executionDType);
    CpuCudaComparisonReport report;
    report.expertSelectionMatches =
        cpuRouting.selectedExpertIds == cudaRouting.selectedExpertIds;
    report.routerLogits = compare(cpuRouterLogits, cudaRouterLogits, tolerance);
    report.routingScores = compare(cpuRouting.routingScores,
                                   cudaRouting.routingScores, tolerance);
    report.gateProjection = compare(cpuTrace.gateProjection,
                                    cudaTrace.gateProjection, tolerance);
    report.upProjection = compare(cpuTrace.upProjection,
                                  cudaTrace.upProjection, tolerance);
    report.activation = compare(cpuTrace.activation, cudaTrace.activation, tolerance);
    report.gated = compare(cpuTrace.gated, cudaTrace.gated, tolerance);
    report.finalOutput = compare(cpuTrace.output, cudaTrace.output, tolerance);
    return report;
}

ExpertCombinationReference CorrectnessOracle::combineExperts(
    std::size_t tokenCount,
    std::size_t hiddenDimension,
    std::span<const RoutedExpertOutput> expertOutputs) {
    if (tokenCount == 0 || hiddenDimension == 0 || expertOutputs.empty() ||
        tokenCount > std::numeric_limits<std::size_t>::max() / hiddenDimension) {
        throw std::invalid_argument("expert combination dimensions are invalid");
    }
    ExpertCombinationReference result;
    result.output.assign(tokenCount * hiddenDimension, 0.0F);
    result.tokenWeightSums.assign(tokenCount, 0.0F);
    for (const auto& expert : expertOutputs) {
        if (expert.tokenIndices.empty() ||
            expert.tokenIndices.size() != expert.routingWeights.size() ||
            expert.tokenIndices.size() >
                std::numeric_limits<std::size_t>::max() / hiddenDimension ||
            expert.values.size() != expert.tokenIndices.size() * hiddenDimension) {
            throw std::invalid_argument("routed expert output metadata is invalid");
        }
        for (std::size_t row = 0; row < expert.tokenIndices.size(); ++row) {
            const auto token = expert.tokenIndices[row];
            const auto weight = expert.routingWeights[row];
            if (token >= tokenCount || !std::isfinite(weight)) {
                throw std::invalid_argument("routed expert assignment is invalid");
            }
            result.tokenWeightSums[token] += weight;
            for (std::size_t hidden = 0; hidden < hiddenDimension; ++hidden) {
                result.output[token * hiddenDimension + hidden] +=
                    weight * expert.values[row * hiddenDimension + hidden];
            }
        }
    }
    return result;
}

ExpertCombinationComparisonReport CorrectnessOracle::compareExpertCombination(
    const router::BatchRouterDecision& actualRouting,
    const router::BatchRouterDecision& expectedRouting,
    std::span<const RoutedExpertOutput> actualExperts,
    std::span<const RoutedExpertOutput> expectedExperts,
    std::span<const float> actualCombined,
    std::span<const float> expectedCombined,
    tensor::DType executionDType) {
    const auto tolerance = toleranceFor(executionDType);
    ExpertCombinationComparisonReport report;
    report.routingMatches = actualRouting.layerId == expectedRouting.layerId &&
                            actualRouting.tokens.size() == expectedRouting.tokens.size();
    if (report.routingMatches) {
        for (std::size_t token = 0; token < actualRouting.tokens.size(); ++token) {
            const auto& actual = actualRouting.tokens[token];
            const auto& expected = expectedRouting.tokens[token];
            if (actual.selectedExpertIds != expected.selectedExpertIds ||
                !compare(actual.routingScores, expected.routingScores,
                         tolerance).matches) {
                report.routingMatches = false;
                break;
            }
        }
    }
    report.normalizedWeights = !actualRouting.tokens.empty();
    for (const auto& token : actualRouting.tokens) {
        const auto sum = std::accumulate(token.routingScores.begin(),
                                         token.routingScores.end(), 0.0F);
        const auto difference = std::abs(sum - 1.0F);
        if (!std::isfinite(sum) ||
            difference > tolerance.absolute + tolerance.relative) {
            report.normalizedWeights = false;
            break;
        }
    }
    if (actualExperts.size() != expectedExperts.size()) {
        report.individualOutputs.push_back({false, 1, 0.0F, 0.0F});
    } else {
        report.individualOutputs.reserve(actualExperts.size());
        for (std::size_t index = 0; index < actualExperts.size(); ++index) {
            const auto& actual = actualExperts[index];
            const auto& expected = expectedExperts[index];
            if (actual.expertId != expected.expertId ||
                actual.tokenIndices != expected.tokenIndices ||
                actual.routingWeights.size() != expected.routingWeights.size() ||
                !compare(actual.routingWeights, expected.routingWeights,
                         tolerance).matches) {
                report.individualOutputs.push_back({false, 1, 0.0F, 0.0F});
            } else {
                report.individualOutputs.push_back(
                    compare(actual.values, expected.values, tolerance));
            }
        }
    }
    report.combinedOutput = compare(actualCombined, expectedCombined, tolerance);
    return report;
}

std::vector<float> CorrectnessOracle::applyRoPE(
    std::span<const float> values,
    std::size_t tokenCount,
    std::size_t headCount,
    std::size_t headDimension,
    std::size_t positionOffset,
    float theta) {
    if (tokenCount == 0 || headCount == 0 || headDimension == 0 ||
        headDimension % 2 != 0 || !std::isfinite(theta) || theta <= 0.0F ||
        headCount > std::numeric_limits<std::size_t>::max() / headDimension ||
        tokenCount > std::numeric_limits<std::size_t>::max() /
                         (headCount * headDimension) ||
        values.size() != tokenCount * headCount * headDimension) {
        throw std::invalid_argument("oracle RoPE dimensions are invalid");
    }
    std::vector<float> result(values.begin(), values.end());
    for (std::size_t token = 0; token < tokenCount; ++token) {
        for (std::size_t head = 0; head < headCount; ++head) {
            const auto base = (token * headCount + head) * headDimension;
            for (std::size_t dimension = 0; dimension < headDimension;
                 dimension += 2) {
                const auto angle = static_cast<double>(positionOffset + token) /
                    std::pow(theta, static_cast<double>(dimension) /
                                         static_cast<double>(headDimension));
                const auto first = result[base + dimension];
                const auto second = result[base + dimension + 1];
                result[base + dimension] = static_cast<float>(
                    first * std::cos(angle) - second * std::sin(angle));
                result[base + dimension + 1] = static_cast<float>(
                    first * std::sin(angle) + second * std::cos(angle));
            }
        }
    }
    return result;
}

std::vector<float> CorrectnessOracle::causalAttention(
    std::span<const float> query,
    std::span<const float> key,
    std::span<const float> value,
    std::span<const std::uint64_t> keyPositions,
    std::size_t queryTokenCount,
    std::size_t headCount,
    std::size_t keyValueHeadCount,
    std::size_t headDimension,
    std::uint64_t queryPositionOffset) {
    if (queryTokenCount == 0 || headCount == 0 || keyValueHeadCount == 0 ||
        headDimension == 0 || headCount % keyValueHeadCount != 0 ||
        query.size() != queryTokenCount * headCount * headDimension ||
        key.size() != keyPositions.size() * keyValueHeadCount * headDimension ||
        value.size() != key.size()) {
        throw std::invalid_argument("oracle attention dimensions are invalid");
    }
    std::vector<float> result(query.size(), 0.0F);
    const auto headsPerKv = headCount / keyValueHeadCount;
    const auto scale = 1.0 / std::sqrt(static_cast<double>(headDimension));
    for (std::size_t token = 0; token < queryTokenCount; ++token) {
        const auto queryPosition = queryPositionOffset + token;
        for (std::size_t head = 0; head < headCount; ++head) {
            const auto kvHead = head / headsPerKv;
            std::vector<double> scores(keyPositions.size(),
                                       -std::numeric_limits<double>::infinity());
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t cached = 0; cached < keyPositions.size(); ++cached) {
                if (keyPositions[cached] > queryPosition) continue;
                double dot{};
                for (std::size_t dimension = 0; dimension < headDimension;
                     ++dimension) {
                    dot += query[(token * headCount + head) * headDimension + dimension] *
                           key[(cached * keyValueHeadCount + kvHead) * headDimension +
                               dimension];
                }
                scores[cached] = dot * scale;
                maximum = std::max(maximum, scores[cached]);
            }
            double denominator{};
            for (auto& score : scores) {
                if (std::isfinite(score)) {
                    score = std::exp(score - maximum);
                    denominator += score;
                } else {
                    score = 0.0;
                }
            }
            if (denominator <= 0.0 || !std::isfinite(denominator)) {
                throw std::runtime_error("oracle causal attention has no visible key");
            }
            for (std::size_t cached = 0; cached < keyPositions.size(); ++cached) {
                const auto probability = scores[cached] / denominator;
                for (std::size_t dimension = 0; dimension < headDimension;
                     ++dimension) {
                    result[(token * headCount + head) * headDimension + dimension] +=
                        static_cast<float>(probability *
                            value[(cached * keyValueHeadCount + kvHead) *
                                      headDimension + dimension]);
                }
            }
        }
    }
    return result;
}

ModelLayerComparisonReport CorrectnessOracle::compareModelLayers(
    std::span<const std::vector<float>> actual,
    std::span<const std::vector<float>> expected,
    tensor::DType executionDType) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::invalid_argument("model layer comparison count is invalid");
    }
    ModelLayerComparisonReport report;
    report.layers.reserve(actual.size());
    const auto tolerance = toleranceFor(executionDType);
    for (std::size_t layer = 0; layer < actual.size(); ++layer) {
        report.layers.push_back(compare(actual[layer], expected[layer], tolerance));
    }
    return report;
}

} // namespace hypermoe::validation
