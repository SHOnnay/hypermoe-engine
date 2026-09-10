#include "validation/RealModelValidation.hpp"

#include "importer/qwen/QwenImporter.hpp"

#include <sstream>
#include <stdexcept>

namespace hypermoe::validation {

bool RealModelComparison::matches() const noexcept {
    return logits.matches && transformer.matches() && experts.matches();
}

std::string RealModelComparison::toJson() const {
    const auto comparison = [](const ComparisonResult& value) {
        std::ostringstream output;
        output << "{\"matches\":" << (value.matches ? "true" : "false")
               << ",\"mismatches\":" << value.mismatchCount
               << ",\"maximum_absolute_error\":" << value.maximumAbsoluteError
               << ",\"maximum_relative_error\":" << value.maximumRelativeError
               << '}';
        return output.str();
    };
    std::ostringstream output;
    output << "{\"matches\":" << (matches() ? "true" : "false")
           << ",\"logits\":" << comparison(logits)
           << ",\"transformer_layers\":" << transformer.layers.size()
           << ",\"experts\":" << experts.layers.size() << '}';
    return output.str();
}

RealModelPreparation RealModelValidator::prepareQwen(
    const std::filesystem::path& artifact,
    const std::filesystem::path& outputDirectory) {
    if (artifact.empty() || outputDirectory.empty()) {
        throw std::invalid_argument(
            "real model preparation requires artifact and output paths");
    }
    RealModelPreparation result;
    const auto importStart = std::chrono::steady_clock::now();
    const auto source = importer::qwen::QwenImporter{}.inspect(artifact);
    result.importTime = std::chrono::steady_clock::now() - importStart;
    const auto validationStart = std::chrono::steady_clock::now();
    result.checkpoint = importer::validation::CheckpointValidator::validate(
        artifact, source);
    result.validationTime = std::chrono::steady_clock::now() - validationStart;
    const auto packingStart = std::chrono::steady_clock::now();
    result.packing = conversion::ExpertPacker{}.pack(
        source, std::filesystem::is_directory(artifact)
                    ? artifact : artifact.parent_path(),
        outputDirectory);
    result.packingTime = std::chrono::steady_clock::now() - packingStart;
    result.manifest = models::ModelManifest::load(outputDirectory / "manifest.json");
    if (!result.packing.validationPassed || !result.manifest.runtimeArchitecture ||
        !result.manifest.modelIO || result.manifest.layers.size() !=
            result.manifest.runtimeArchitecture->layerCount) {
        throw std::runtime_error(
            "Qwen artifact is valid for packing but lacks a complete forward manifest");
    }
    return result;
}

RealModelComparison RealModelValidator::compare(
    const RealModelTrace& cpu, const RealModelTrace& cuda,
    tensor::DType executionDType) {
    if (cpu.transformerOutputs.empty() || cpu.expertOutputs.empty() ||
        cpu.transformerOutputs.size() != cuda.transformerOutputs.size() ||
        cpu.expertOutputs.size() != cuda.expertOutputs.size()) {
        throw std::invalid_argument(
            "real model traces require matching nonempty layer and expert outputs");
    }
    const auto tolerance = CorrectnessOracle::toleranceFor(executionDType);
    RealModelComparison result;
    result.logits = CorrectnessOracle::compare(cuda.logits, cpu.logits, tolerance);
    result.transformer = CorrectnessOracle::compareModelLayers(
        cuda.transformerOutputs, cpu.transformerOutputs, executionDType);
    result.experts = CorrectnessOracle::compareModelLayers(
        cuda.expertOutputs, cpu.expertOutputs, executionDType);
    return result;
}

} // namespace hypermoe::validation
