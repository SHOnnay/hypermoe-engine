#include "models/runtime/PackedModelRuntime.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
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
        text.remove_prefix(separator + 1U);
    }
    if (result.empty()) throw std::invalid_argument("token sequence is empty");
    return result;
}

struct Measurement {
    double latencyMs{};
    std::uint64_t transfers{};
    std::uint64_t transferBytes{};
    double cacheHitRate{};
    std::uint64_t usefulPrefetches{};
    std::uint64_t wastedPrefetches{};
    std::uint64_t synchronizations{};
};

Measurement measure(const std::filesystem::path& artifact,
                    std::span<const std::uint32_t> tokens,
                    hypermoe::tensor::Device device,
                    bool adaptive) {
    hypermoe::models::runtime::PackedRuntimeConfiguration configuration;
    configuration.device = device;
    configuration.adaptivePrediction = adaptive;
    configuration.adaptiveResidency = adaptive;
    hypermoe::models::runtime::PackedModelRuntime runtime(artifact, configuration);
    auto cache = runtime.createKVCache(tokens.size());
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        const std::span<const std::uint32_t> token{tokens.data() + position, 1};
        (void)runtime.forward(token, position, *cache);
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto snapshot = runtime.snapshot();
    return {std::chrono::duration<double, std::milli>(elapsed).count(),
            snapshot.profiler.nvmeReads, snapshot.profiler.nvmeBytes,
            snapshot.profiler.cacheHitRate(), snapshot.profiler.prefetchUseful,
            snapshot.profiler.prefetchWasted,
            snapshot.profiler.synchronizationCount};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "usage: hypermoe_runtime_optimization_benchmark "
                         "<runtime-artifact> <token-ids-csv> <cpu|cuda> [report.json]\n";
            return 2;
        }
        const auto tokens = parseTokens(argv[2]);
        const std::string_view deviceName = argv[3];
        const auto device = deviceName == "cuda"
            ? hypermoe::tensor::Device::cuda()
            : hypermoe::tensor::Device::cpu();
        if (deviceName != "cpu" && deviceName != "cuda") {
            throw std::invalid_argument("benchmark device must be cpu or cuda");
        }
        const auto baseline = measure(argv[1], tokens, device, false);
        const auto adaptive = measure(argv[1], tokens, device, true);
        const auto latencyImprovement = baseline.latencyMs == 0.0
            ? 0.0
            : 100.0 * (baseline.latencyMs - adaptive.latencyMs) /
                  baseline.latencyMs;
        const auto transferReduction = baseline.transferBytes == 0
            ? 0.0
            : 100.0 * static_cast<double>(baseline.transferBytes -
                                           std::min(baseline.transferBytes,
                                                    adaptive.transferBytes)) /
                  static_cast<double>(baseline.transferBytes);
        std::ostringstream json;
        const auto writeMeasurement = [&](std::string_view name,
                                          const Measurement& value) {
            json << "  \"" << name << "\": {\n"
                 << "    \"latency_ms\": " << value.latencyMs << ",\n"
                 << "    \"nvme_reads\": " << value.transfers << ",\n"
                 << "    \"transfer_bytes\": " << value.transferBytes << ",\n"
                 << "    \"cache_hit_rate\": " << value.cacheHitRate << ",\n"
                 << "    \"useful_prefetches\": " << value.usefulPrefetches << ",\n"
                 << "    \"wasted_prefetches\": " << value.wastedPrefetches << ",\n"
                 << "    \"synchronizations\": " << value.synchronizations << "\n"
                 << "  }";
        };
        json << std::fixed << std::setprecision(6)
             << "{\n  \"schema\": \"hypermoe.runtime-optimization.v1\",\n"
             << "  \"tokens\": " << tokens.size() << ",\n";
        writeMeasurement("baseline", baseline);
        json << ",\n";
        writeMeasurement("adaptive", adaptive);
        json << ",\n  \"latency_improvement_percent\": " << latencyImprovement
             << ",\n  \"transfer_reduction_percent\": " << transferReduction
             << "\n}\n";
        const std::filesystem::path report =
            argc == 5 ? argv[4] : "runtime_optimization_report.json";
        std::ofstream output(report, std::ios::binary | std::ios::trunc);
        output << json.str();
        if (!output) throw std::runtime_error("cannot write optimization report");
        std::cout << json.str();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "runtime optimization benchmark failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
