#pragma once

#include "importer/validation/CheckpointValidator.hpp"
#include "models/ModelManifest.hpp"
#include "tools/model_convert/ExpertPacker.hpp"
#include "validation/CorrectnessOracle.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace hypermoe::models::runtime {
class PackedModelRuntime;
struct ModelForwardResult;
}

namespace hypermoe::validation {

struct RealModelTrace {
    std::vector<float> embeddings;
    std::vector<std::vector<float>> attentionOutputs;
    std::vector<float> logits;
    std::vector<float> finalNormalization;
    std::vector<std::vector<float>> transformerOutputs;
    std::vector<std::vector<float>> expertOutputs;
    std::vector<std::vector<ExpertId>> selectedExperts;
};

struct RealModelComparison {
    ComparisonResult embeddings;
    ModelLayerComparisonReport attention;
    ComparisonResult logits;
    ComparisonResult finalNormalization;
    ModelLayerComparisonReport transformer;
    ModelLayerComparisonReport experts;
    bool routingMatches{};

    [[nodiscard]] bool matches() const noexcept;
    [[nodiscard]] std::string toJson() const;
};

struct RealExecutionValidationReport {
    bool cudaAvailable{};
    bool executed{};
    std::string message;
    RealModelComparison comparison;
    std::chrono::nanoseconds cpuTime{};
    std::chrono::nanoseconds cudaTime{};

    [[nodiscard]] bool matches() const noexcept;
    [[nodiscard]] std::string toJson() const;
};

struct RealModelPreparation {
    models::ModelManifest manifest;
    importer::validation::CheckpointValidationReport checkpoint;
    conversion::PackingReport packing;
    std::chrono::nanoseconds importTime{};
    std::chrono::nanoseconds validationTime{};
    std::chrono::nanoseconds packingTime{};
};

class RealModelValidator {
public:
    [[nodiscard]] static RealModelPreparation prepareQwen(
        const std::filesystem::path& artifact,
        const std::filesystem::path& outputDirectory);
    [[nodiscard]] static RealModelComparison compare(
        const RealModelTrace& cpu, const RealModelTrace& cuda,
        tensor::DType executionDType = tensor::DType::FP32);
    [[nodiscard]] static RealModelTrace capture(
        const models::runtime::PackedModelRuntime& runtime,
        const models::runtime::ModelForwardResult& result);
    [[nodiscard]] static RealExecutionValidationReport validateCpuCuda(
        const std::filesystem::path& runtimeArtifact,
        std::span<const std::uint32_t> tokenIds,
        int cudaDevice = 0);
};

} // namespace hypermoe::validation
