#include "models/ModelManifest.hpp"
#include "models/runtime/PackedModelRuntime.hpp"
#include "profiling/RealModelProfile.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::uint32_t> parseTokens(std::string_view text) {
    std::vector<std::uint32_t> result;
    while (!text.empty()) {
        const auto separator = text.find(',');
        const auto field = text.substr(0, separator);
        std::uint64_t value{};
        const auto [end, error] = std::from_chars(
            field.data(), field.data() + field.size(), value);
        if (field.empty() || error != std::errc{} ||
            end != field.data() + field.size() ||
            value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("token IDs must be comma-separated uint32 values");
        }
        result.push_back(static_cast<std::uint32_t>(value));
        if (separator == std::string_view::npos) break;
        text.remove_prefix(separator + 1);
    }
    if (result.empty()) throw std::invalid_argument("at least one token ID is required");
    return result;
}

std::uint64_t parameterCount(const hypermoe::models::ModelManifest& manifest) {
    if (manifest.parameterCount != 0) return manifest.parameterCount;
    std::uint64_t result{};
    for (const auto& tensor : manifest.tensors) {
        const auto elements = tensor.shape.elementCount();
        if (elements > std::numeric_limits<std::uint64_t>::max() - result) {
            throw std::overflow_error("model parameter count overflows");
        }
        result += elements;
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: hypermoe_profile_real_model <runtime-artifact> "
                         "<token-ids-csv> <cpu|cuda> [report.json] "
                         "[--expert-device-budget 512MiB] "
                         "[--expert-ram-budget 2GiB] [--overlap on|off]\n";
            return 2;
        }
        const std::filesystem::path artifact = argv[1];
        const auto tokenIds = parseTokens(argv[2]);
        const std::string_view deviceName = argv[3];
        hypermoe::models::runtime::PackedRuntimeConfiguration configuration;
        if (deviceName == "cuda") {
            configuration.device = hypermoe::tensor::Device::cuda();
        } else if (deviceName != "cpu") {
            throw std::invalid_argument("benchmark device must be cpu or cuda");
        }
        std::filesystem::path reportPath{"real_model_profile.json"};
        bool reportSpecified{};
        for (int index = 4; index < argc; ++index) {
            const std::string_view option{argv[index]};
            if (option == "--expert-device-budget" || option == "--expert-ram-budget" ||
                option == "--overlap") {
                if (++index >= argc) throw std::invalid_argument("benchmark option requires a value");
                const std::string_view value{argv[index]};
                if (option == "--overlap") {
                    if (value != "on" && value != "off") throw std::invalid_argument("overlap must be on or off");
                    configuration.transferComputeOverlap = value == "on";
                } else {
                    const auto bytes = decltype(configuration)::parseBudgetBytes(value);
                    if (option == "--expert-device-budget") configuration.expertDeviceBudgetBytes = bytes;
                    else configuration.expertRamBudgetBytes = bytes;
                }
            } else if (!option.starts_with("--") && !reportSpecified) {
                reportPath = argv[index];
                reportSpecified = true;
            } else {
                throw std::invalid_argument("unknown benchmark option or extra report path");
            }
        }
        configuration.validate();
        const auto loadingStarted = std::chrono::steady_clock::now();
        hypermoe::models::runtime::PackedModelRuntime runtime(
            artifact, configuration);
        const auto loadingTime = std::chrono::steady_clock::now() - loadingStarted;
        auto cache = runtime.createKVCache(tokenIds.size());
        std::vector<hypermoe::models::runtime::ModelForwardResult> forwards;
        forwards.reserve(tokenIds.size());
        const auto executionStarted = std::chrono::steady_clock::now();
        for (std::size_t position = 0; position < tokenIds.size(); ++position) {
            const std::span<const std::uint32_t> token{tokenIds.data() + position, 1};
            forwards.push_back(runtime.forward(token, position, *cache));
        }
        const auto wallTime = std::chrono::steady_clock::now() - executionStarted;
        auto profile = hypermoe::profiling::RealModelProfileCollector::collect(
            runtime.manifest(), parameterCount(runtime.manifest()), runtime.device(),
            forwards, runtime.snapshot(), cache->memoryUsageBytes(), wallTime);
        profile.modelLoadingTime = loadingTime;
        profile.transferComputeOverlap = configuration.transferComputeOverlap;
        std::ofstream output(reportPath, std::ios::binary | std::ios::trunc);
        const auto json = profile.toJson();
        output.write(json.data(), static_cast<std::streamsize>(json.size()));
        if (!output) throw std::runtime_error("cannot write real-model profile");
        std::cout << json;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "real-model profile failed: " << error.what() << '\n';
        return 1;
    }
}
