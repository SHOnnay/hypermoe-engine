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

    [[nodiscard]] static std::vector<float> expertMlp(
        std::span<const float> input,
        std::size_t tokens,
        std::size_t hiddenSize,
        std::span<const float> gateInputOutput,
        std::span<const float> upInputOutput,
        std::span<const float> downInputOutput,
        std::size_t intermediateSize);
};

} // namespace hypermoe::validation
