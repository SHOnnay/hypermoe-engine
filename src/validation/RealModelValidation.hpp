#pragma once

#include "importer/validation/CheckpointValidator.hpp"
#include "models/ModelManifest.hpp"
#include "tools/model_convert/ExpertPacker.hpp"
#include "validation/CorrectnessOracle.hpp"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace hypermoe::validation {

struct RealModelTrace {
    std::vector<float> logits;
    std::vector<std::vector<float>> transformerOutputs;
    std::vector<std::vector<float>> expertOutputs;
};

struct RealModelComparison {
    ComparisonResult logits;
    ModelLayerComparisonReport transformer;
    ModelLayerComparisonReport experts;

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
};

} // namespace hypermoe::validation
