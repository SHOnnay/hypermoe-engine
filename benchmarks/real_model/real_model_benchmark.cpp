#include "hardware/HardwareInfo.hpp"
#include "validation/RealModelValidation.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

double milliseconds(std::chrono::nanoseconds duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

std::string nullable(const std::optional<double>& value) {
    if (!value) return "null";
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << *value;
    return output.str();
}

std::string jsonString(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const auto character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    output << '"';
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            std::cerr << "usage: hypermoe_real_model_benchmark <checkpoint> "
                         "<packed-output> [report.json]\n";
            return 2;
        }
        const std::filesystem::path checkpoint = argv[1];
        const std::filesystem::path packed = argv[2];
        const std::filesystem::path reportPath =
            argc == 4 ? argv[3] : "real_model_report.json";
        const auto prepared = hypermoe::validation::RealModelValidator::prepareQwen(
            checkpoint, packed);
        const auto hardware = hypermoe::hardware::detectHardware(
            std::filesystem::is_directory(checkpoint)
                ? checkpoint : checkpoint.parent_path());

        // Forward execution is deliberately a separate validation step: it
        // requires caller-supplied token IDs and a runtime precision policy.
        // Unmeasured fields remain null rather than fabricating model results.
        const std::optional<double> unmeasured;
        std::ofstream output(reportPath, std::ios::trunc);
        output << std::fixed << std::setprecision(6)
               << "{\n  \"schema\": \"hypermoe.real-model-benchmark.v1\",\n"
               << "  \"checkpoint\": "
               << jsonString(checkpoint.generic_string()) << ",\n"
               << "  \"model\": " << jsonString(prepared.manifest.modelName) << ",\n"
               << "  \"architecture\": "
               << jsonString(prepared.manifest.sourceArchitecture) << ",\n"
               << "  \"device\": " << jsonString(hardware.gpuName) << ",\n"
               << "  \"checkpoint_validation\": true,\n"
               << "  \"cuda_available\": "
               << (hardware.cudaAvailable ? "true" : "false") << ",\n"
               << "  \"model_loading_ms\": "
               << milliseconds(prepared.importTime + prepared.validationTime) << ",\n"
               << "  \"packing_ms\": " << milliseconds(prepared.packingTime) << ",\n"
               << "  \"packed_bytes\": " << prepared.packing.bytesWritten << ",\n"
               << "  \"experts\": " << prepared.packing.experts << ",\n"
               << "  \"vram_usage_bytes\": " << nullable(unmeasured) << ",\n"
               << "  \"ram_usage_bytes\": " << nullable(unmeasured) << ",\n"
               << "  \"first_token_latency_ms\": " << nullable(unmeasured) << ",\n"
               << "  \"generation_tokens_per_second\": " << nullable(unmeasured) << ",\n"
               << "  \"expert_transfer_bytes\": " << nullable(unmeasured) << ",\n"
               << "  \"cache_hit_rate\": " << nullable(unmeasured) << "\n}\n";
        if (!output) throw std::runtime_error("cannot write real model report");
        std::cout << reportPath << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "real model benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
