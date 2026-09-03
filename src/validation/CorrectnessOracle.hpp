#pragma once

#include "router/RouterConfig.hpp"
#include "router/RouterDecision.hpp"
#include "tensor/DType.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace hypermoe::validation {

struct NumericalTolerance {
    float absolute{};
    float relative{};
};

struct ComparisonResult {
    bool matches{};
    std::size_t mismatchCount{};
    float maximumAbsoluteError{};
    float maximumRelativeError{};
};

struct ExpertOracleTrace {
    std::vector<float> gateProjection;
    std::vector<float> upProjection;
    std::vector<float> activation;
    std::vector<float> gated;
    std::vector<float> output;
};

struct CpuCudaComparisonReport {
    bool expertSelectionMatches{};
    ComparisonResult routerLogits;
    ComparisonResult routingScores;
    ComparisonResult gateProjection;
    ComparisonResult upProjection;
    ComparisonResult activation;
    ComparisonResult gated;
    ComparisonResult finalOutput;

    [[nodiscard]] bool matches() const noexcept;
};

struct RoutedExpertOutput {
    ExpertId expertId{};
    std::vector<std::size_t> tokenIndices;
    std::vector<float> routingWeights;
    std::vector<float> values;
};

struct ExpertCombinationReference {
    std::vector<float> output;
    std::vector<float> tokenWeightSums;
};

struct ExpertCombinationComparisonReport {
    bool routingMatches{};
    bool normalizedWeights{};
    std::vector<ComparisonResult> individualOutputs;
    ComparisonResult combinedOutput;

    [[nodiscard]] bool matches() const noexcept;
};

class CorrectnessOracle {
public:
    [[nodiscard]] static NumericalTolerance toleranceFor(tensor::DType dtype) noexcept;
    [[nodiscard]] static ComparisonResult compare(
        std::span<const float> actual,
        std::span<const float> expected,
        NumericalTolerance tolerance);

    [[nodiscard]] static router::RouterDecision route(
        LayerId layerId,
        std::span<const float> hidden,
        std::span<const float> inputOutputWeights,
        const router::RouterConfig& config);
    [[nodiscard]] static std::vector<float> routerLogits(
        std::span<const float> hidden,
        std::span<const float> inputOutputWeights,
        std::size_t expertCount);
    [[nodiscard]] static router::RouterDecision selectFromLogits(
        LayerId layerId,
        std::span<const float> logits,
        const router::RouterConfig& config);

    [[nodiscard]] static std::vector<float> expertMlp(
        std::span<const float> input,
        std::size_t tokens,
        std::size_t hiddenSize,
        std::span<const float> gateInputOutput,
        std::span<const float> upInputOutput,
        std::span<const float> downInputOutput,
        std::size_t intermediateSize);
    [[nodiscard]] static ExpertOracleTrace expertMlpTrace(
        std::span<const float> input,
        std::size_t tokens,
        std::size_t hiddenSize,
        std::span<const float> gateInputOutput,
        std::span<const float> upInputOutput,
        std::span<const float> downInputOutput,
        std::size_t intermediateSize);
    [[nodiscard]] static CpuCudaComparisonReport compareCpuCuda(
        std::span<const float> cpuRouterLogits,
        std::span<const float> cudaRouterLogits,
        const router::RouterDecision& cpuRouting,
        const router::RouterDecision& cudaRouting,
        const ExpertOracleTrace& cpuTrace,
        const ExpertOracleTrace& cudaTrace,
        tensor::DType executionDType = tensor::DType::FP32);
    [[nodiscard]] static ExpertCombinationReference combineExperts(
        std::size_t tokenCount,
        std::size_t hiddenDimension,
        std::span<const RoutedExpertOutput> expertOutputs);
    [[nodiscard]] static ExpertCombinationComparisonReport compareExpertCombination(
        const router::BatchRouterDecision& actualRouting,
        const router::BatchRouterDecision& expectedRouting,
        std::span<const RoutedExpertOutput> actualExperts,
        std::span<const RoutedExpertOutput> expectedExperts,
        std::span<const float> actualCombined,
        std::span<const float> expectedCombined,
        tensor::DType executionDType = tensor::DType::FP32);
};

} // namespace hypermoe::validation
