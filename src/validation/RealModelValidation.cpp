#include "validation/RealModelValidation.hpp"

#include "importer/qwen/QwenImporter.hpp"
#include "models/runtime/PackedModelRuntime.hpp"
#include "backend/CudaBackend.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace hypermoe::validation {

bool RealModelComparison::matches() const noexcept {
    return embeddings.matches && attention.matches() && logits.matches &&
           finalNormalization.matches && transformer.matches() && experts.matches() &&
           routingMatches;
}

bool RealExecutionValidationReport::matches() const noexcept {
    return executed && comparison.matches();
}

std::string RealExecutionValidationReport::toJson() const {
    std::ostringstream output;
    output << "{\"schema\":\"hypermoe.real-execution-validation.v1\""
           << ",\"cuda_available\":" << (cudaAvailable ? "true" : "false")
           << ",\"executed\":" << (executed ? "true" : "false")
           << ",\"matches\":" << (matches() ? "true" : "false")
           << ",\"cpu_ms\":"
           << std::chrono::duration<double, std::milli>(cpuTime).count()
           << ",\"cuda_ms\":";
    if (executed) {
        output << std::chrono::duration<double, std::milli>(cudaTime).count()
               << ",\"comparison\":" << comparison.toJson();
    } else {
        output << "null,\"message\":\"" << message << '"';
    }
    output << "}\n";
    return output.str();
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
    const auto summarize = [](const ModelLayerComparisonReport& report) {
        ComparisonResult result{!report.layers.empty(), 0, 0.0F, 0.0F};
        for (const auto& layer : report.layers) {
            result.matches = result.matches && layer.matches;
            result.mismatchCount += layer.mismatchCount;
            result.maximumAbsoluteError = std::max(
                result.maximumAbsoluteError, layer.maximumAbsoluteError);
            result.maximumRelativeError = std::max(
                result.maximumRelativeError, layer.maximumRelativeError);
        }
        return result;
    };
    std::ostringstream output;
    output << "{\"matches\":" << (matches() ? "true" : "false")
           << ",\"embeddings\":" << comparison(embeddings)
           << ",\"routing_matches\":" << (routingMatches ? "true" : "false")
           << ",\"attention\":" << comparison(summarize(attention))
           << ",\"attention_layers\":" << attention.layers.size()
           << ",\"logits\":" << comparison(logits)
           << ",\"final_normalization\":" << comparison(finalNormalization)
           << ",\"transformer\":" << comparison(summarize(transformer))
           << ",\"transformer_layers\":" << transformer.layers.size()
           << ",\"expert_outputs\":" << comparison(summarize(experts))
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
    if (cpu.attentionOutputs.empty() || cpu.transformerOutputs.empty() ||
        cpu.expertOutputs.empty() ||
        cpu.attentionOutputs.size() != cuda.attentionOutputs.size() ||
        cpu.transformerOutputs.size() != cuda.transformerOutputs.size() ||
        cpu.expertOutputs.size() != cuda.expertOutputs.size()) {
        throw std::invalid_argument(
            "real model traces require matching nonempty layer and expert outputs");
    }
    const auto tolerance = CorrectnessOracle::toleranceFor(executionDType);
    RealModelComparison result;
    result.embeddings = CorrectnessOracle::compare(
        cuda.embeddings, cpu.embeddings, tolerance);
    result.attention = CorrectnessOracle::compareModelLayers(
        cuda.attentionOutputs, cpu.attentionOutputs, executionDType);
    result.logits = CorrectnessOracle::compare(cuda.logits, cpu.logits, tolerance);
    result.finalNormalization = CorrectnessOracle::compare(
        cuda.finalNormalization, cpu.finalNormalization, tolerance);
    result.transformer = CorrectnessOracle::compareModelLayers(
        cuda.transformerOutputs, cpu.transformerOutputs, executionDType);
    result.experts = CorrectnessOracle::compareModelLayers(
        cuda.expertOutputs, cpu.expertOutputs, executionDType);
    result.routingMatches = cuda.selectedExperts == cpu.selectedExperts;
    return result;
}

RealModelTrace RealModelValidator::capture(
    const models::runtime::PackedModelRuntime& runtime,
    const models::runtime::ModelForwardResult& result) {
    const auto values = [&](tensor::TensorView tensor) {
        auto host = runtime.materializeHost(tensor);
        const auto* begin = static_cast<const float*>(host.data());
        return std::vector<float>{begin, begin + host.shape().elementCount()};
    };
    RealModelTrace trace;
    trace.embeddings = values(result.embeddings.view());
    trace.finalNormalization = values(result.normalizedHiddenStates.view());
    trace.logits = values(result.logits.view());
    for (const auto& layer : result.transformer.layers) {
        trace.attentionOutputs.push_back(values(layer.attentionOutput.view()));
        trace.transformerOutputs.push_back(values(layer.output.view()));
        for (const auto& expert : layer.expertOutputs) {
            trace.expertOutputs.push_back(values(expert.view()));
        }
        for (const auto& decision : layer.routing) {
            trace.selectedExperts.push_back(decision.selectedExpertIds);
        }
    }
    return trace;
}

RealExecutionValidationReport RealModelValidator::validateCpuCuda(
    const std::filesystem::path& runtimeArtifact,
    std::span<const std::uint32_t> tokenIds,
    int cudaDevice) {
    if (tokenIds.empty() || cudaDevice < 0) {
        throw std::invalid_argument("CPU/CUDA validation input is invalid");
    }
    RealExecutionValidationReport report;
    RealModelTrace cpuTrace;
    {
        models::runtime::PackedModelRuntime cpu(runtimeArtifact);
        const auto started = std::chrono::steady_clock::now();
        const auto forward = cpu.forward(tokenIds);
        report.cpuTime = std::chrono::steady_clock::now() - started;
        cpuTrace = capture(cpu, forward);
    }
    const auto cuda = backend::CudaBackend::query(cudaDevice);
    report.cudaAvailable = cuda.available;
    if (!cuda.available) {
        report.message = "CUDA runtime/device unavailable; CPU reference completed";
        return report;
    }
    models::runtime::PackedRuntimeConfiguration configuration;
    configuration.device = tensor::Device::cuda(cudaDevice);
    models::runtime::PackedModelRuntime gpu(runtimeArtifact, configuration);
    const auto started = std::chrono::steady_clock::now();
    const auto forward = gpu.forward(tokenIds);
    report.cudaTime = std::chrono::steady_clock::now() - started;
    const auto cudaTrace = capture(gpu, forward);
    report.comparison = compare(cpuTrace, cudaTrace, tensor::DType::FP32);
    report.executed = true;
    report.message = report.comparison.matches()
        ? "CPU and CUDA traces match" : "CPU and CUDA traces diverge";
    return report;
}

} // namespace hypermoe::validation
