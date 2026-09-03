#include "backend/cuda/CudaRuntimeValidator.hpp"

#include "backend/cuda/CudaRuntime.hpp"
#include "backend/cuda/CudaStreamManager.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hypermoe::backend {
namespace {

std::string jsonEscape(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

void addCheck(CudaRuntimeValidationReport& report,
              std::string name,
              bool passed,
              std::string detail) {
    report.checks.push_back({std::move(name), passed, std::move(detail)});
}

} // namespace

std::string_view toString(CudaValidationStatus status) noexcept {
    switch (status) {
    case CudaValidationStatus::Passed: return "PASSED";
    case CudaValidationStatus::Skipped: return "SKIPPED";
    case CudaValidationStatus::Failed: return "FAILED";
    }
    return "UNKNOWN";
}

bool CudaRuntimeValidationReport::passed() const noexcept {
    return status == CudaValidationStatus::Passed;
}

bool CudaRuntimeValidationReport::skipped() const noexcept {
    return status == CudaValidationStatus::Skipped;
}

std::string CudaRuntimeValidationReport::toJson() const {
    std::ostringstream output;
    output << "{\n"
           << "  \"schema\": \"hypermoe.cuda-validation.v1\",\n"
           << "  \"status\": \"" << toString(status) << "\",\n"
           << "  \"cuda_compiled\": " << (compiledWithCuda ? "true" : "false")
           << ",\n"
           << "  \"cuda_available\": " << (runtimeAvailable ? "true" : "false")
           << ",\n"
           << "  \"message\": \"" << jsonEscape(message) << "\",\n"
           << "  \"device\": " << device.toJson() << ",\n"
           << "  \"checks\": [";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        const auto& check = checks[index];
        output << (index == 0 ? "\n" : ",\n")
               << "    {\"name\":\"" << jsonEscape(check.name)
               << "\",\"passed\":" << (check.passed ? "true" : "false")
               << ",\"detail\":\"" << jsonEscape(check.detail) << "\"}";
    }
    if (!checks.empty()) output << '\n';
    output << "  ]\n}\n";
    return output.str();
}

void CudaRuntimeValidationReport::writeJson(
    const std::filesystem::path& path) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create CUDA validation report: " +
                                 path.string());
    }
    output << toJson();
    if (!output) {
        throw std::runtime_error("failed writing CUDA validation report: " +
                                 path.string());
    }
}

CudaRuntimeValidationReport CudaRuntimeValidator::validate(int device) noexcept {
    CudaRuntimeValidationReport report;
    report.compiledWithCuda = CudaRuntime::compiledWithCuda();
    try {
        auto runtime = std::make_shared<CudaRuntime>(device);
        report.runtimeAvailable = runtime->available();
        report.device = runtime->deviceInfo();
        if (!report.runtimeAvailable) {
            report.status = CudaValidationStatus::Skipped;
            report.message = report.compiledWithCuda
                                 ? "CUDA was compiled but no usable NVIDIA device was found"
                                 : "CUDA toolkit support was not compiled";
            return report;
        }

        addCheck(report, "device_properties", !report.device.name.empty(),
                 report.device.name);
        addCheck(report, "compute_capability",
                 report.device.computeCapabilityMajor > 0,
                 report.device.computeCapability());
        addCheck(report, "vram",
                 report.device.totalVramBytes > 0 &&
                     report.device.freeVramBytes <= report.device.totalVramBytes,
                 std::to_string(report.device.freeVramBytes) + " free of " +
                     std::to_string(report.device.totalVramBytes) + " bytes");
        addCheck(report, "cuda_versions",
                 report.device.runtimeVersion > 0 && report.device.driverVersion > 0,
                 "runtime=" + std::to_string(report.device.runtimeVersion) +
                     ", driver=" + std::to_string(report.device.driverVersion));

        CudaStreamManager streams(runtime);
        report.device = runtime->deviceInfo();
        addCheck(report, "streams", streams.available() &&
                     report.device.streamCount == 3,
                 std::to_string(report.device.streamCount) + " managed streams");

        const auto start = runtime->createEvent(true);
        EventHandle end = nullptr;
        try {
            end = runtime->createEvent(true);
            const auto compute = streams.stream(CudaStreamRole::Compute);
            runtime->recordEvent(start, compute);
            runtime->recordEvent(end, compute);
            runtime->synchronizeEvent(end);
            const auto elapsed = runtime->elapsedMilliseconds(start, end);
            addCheck(report, "events", elapsed >= 0.0F,
                     std::to_string(elapsed) + " ms empty-stream interval");
            runtime->destroyEvent(start);
            runtime->destroyEvent(end);
        } catch (...) {
            runtime->destroyEvent(start);
            runtime->destroyEvent(end);
            throw;
        }

        const bool allPassed = [&] {
            for (const auto& check : report.checks) {
                if (!check.passed) return false;
            }
            return true;
        }();
        report.status = allPassed ? CudaValidationStatus::Passed
                                  : CudaValidationStatus::Failed;
        report.message = allPassed ? "CUDA runtime validation passed"
                                   : "one or more CUDA runtime checks failed";
    } catch (const std::exception& error) {
        report.status = CudaValidationStatus::Failed;
        report.message = error.what();
    } catch (...) {
        report.status = CudaValidationStatus::Failed;
        report.message = "unknown CUDA validation failure";
    }
    return report;
}

CudaRuntimeValidationReport CudaRuntimeValidator::validateAndWrite(
    const std::filesystem::path& reportPath,
    int device) {
    auto report = validate(device);
    report.writeJson(reportPath);
    return report;
}

} // namespace hypermoe::backend
