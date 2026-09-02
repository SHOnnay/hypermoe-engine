#include "profiling/Profiler.hpp"
#include "scheduler/Prefetcher.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::uint32_t kLayers = 8;
constexpr std::uint32_t kExpertsPerLayer = 16;
constexpr std::uint32_t kTopK = 3;
constexpr double kGpuComputeMs = 1.50;
constexpr double kNvmeLoadMs = 3.00;
constexpr double kRamTransferMs = 0.35;
constexpr std::size_t kVramExpertCapacity = 18;

struct Options {
    std::uint32_t tokens{10'000};
    std::uint32_t seed{20260902};
    std::filesystem::path report{"pipeline_report.json"};
};

struct ModeResult {
    double totalLatencyMs{};
    double stallTimeMs{};
    std::uint64_t stallCount{};
    std::uint64_t peakQueueDepth{};
};

struct Comparison {
    ModeResult synchronous;
    ModeResult asynchronous;
    double totalTransferMs{};
    double hiddenTransferMs{};
    std::uint64_t prefetchRequests{};
    std::uint64_t prefetchHits{};
    std::uint64_t prefetchMisses{};
};

template <typename T>
T parseNumber(std::string_view value, std::string_view name) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid value for " + std::string(name));
    }
    return result;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto equals = argument.find('=');
        if (equals == std::string_view::npos) throw std::invalid_argument("expected --name=value");
        const auto name = argument.substr(0, equals);
        const auto value = argument.substr(equals + 1);
        if (name == "--tokens") options.tokens = parseNumber<std::uint32_t>(value, name);
        else if (name == "--seed") options.seed = parseNumber<std::uint32_t>(value, name);
        else if (name == "--report") options.report = value;
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    return options;
}

class LruResidency {
public:
    bool contains(hypermoe::ExpertId id) const { return positions_.contains(id); }

    void touch(hypermoe::ExpertId id) {
        const auto existing = positions_.find(id);
        if (existing != positions_.end()) order_.erase(existing->second);
        order_.push_front(id);
        positions_[id] = order_.begin();
        if (positions_.size() > kVramExpertCapacity) {
            positions_.erase(order_.back());
            order_.pop_back();
        }
    }

private:
    std::list<hypermoe::ExpertId> order_;
    std::unordered_map<hypermoe::ExpertId,
                       std::list<hypermoe::ExpertId>::iterator>
        positions_;
};

std::vector<std::array<hypermoe::ExpertId, kTopK>>
makeRoutes(std::uint32_t tokens, std::uint32_t seed) {
    std::mt19937 random(seed);
    std::array<std::uint32_t, kLayers> previous{};
    std::vector<std::array<hypermoe::ExpertId, kTopK>> routes;
    routes.reserve(static_cast<std::size_t>(tokens) * kLayers);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            std::bernoulli_distribution retain(0.88);
            std::uniform_int_distribution<std::uint32_t> select(0, kExpertsPerLayer - 1);
            const auto primary = retain(random) ? previous[layer] : select(random);
            previous[layer] = primary;
            const auto base = layer * kExpertsPerLayer;
            routes.push_back({base + primary,
                              base + (primary + 3U) % kExpertsPerLayer,
                              base + (primary + 9U) % kExpertsPerLayer});
        }
    }
    return routes;
}

Comparison simulate(const Options& options) {
    const auto routes = makeRoutes(options.tokens, options.seed);
    hypermoe::scheduler::LocalityPrefetcher predictor(kTopK);
    LruResidency residency;
    std::unordered_set<hypermoe::ExpertId> seen;
    std::vector<hypermoe::ExpertId> previousExperts;
    Comparison result;

    for (std::size_t stage = 0; stage < routes.size(); ++stage) {
        const auto& experts = routes[stage];
        double transferMs = 0.0;
        std::uint64_t misses = 0;
        for (const auto expert : experts) {
            if (!residency.contains(expert)) {
                transferMs += seen.contains(expert)
                                  ? kRamTransferMs
                                  : kNvmeLoadMs + kRamTransferMs;
                ++misses;
            }
        }
        result.totalTransferMs += transferMs;
        result.synchronous.totalLatencyMs += transferMs + kGpuComputeMs;
        result.synchronous.stallTimeMs += transferMs;
        if (transferMs > 0.0) ++result.synchronous.stallCount;
        result.synchronous.peakQueueDepth =
            std::max(result.synchronous.peakQueueDepth, misses);

        double predictedMissTransferMs = 0.0;
        if (stage > 0) {
            hypermoe::scheduler::PredictionInput input;
            input.currentLayer = static_cast<hypermoe::LayerId>((stage - 1) % kLayers);
            input.recentExperts = previousExperts;
            const auto currentLayer = static_cast<hypermoe::LayerId>(stage % kLayers);
            auto pattern = std::vector<hypermoe::ExpertId>(experts.begin(), experts.end());
            if (stage % 5 == 0) {
                pattern.back() = currentLayer * kExpertsPerLayer +
                                 static_cast<hypermoe::ExpertId>((stage / 5) % kExpertsPerLayer);
            }
            input.workloadPattern[currentLayer] = std::move(pattern);
            const auto predictions = predictor.predict(input);
            std::unordered_set<hypermoe::ExpertId> predicted;
            for (const auto& prediction : predictions) predicted.insert(prediction.expertId);
            result.prefetchRequests += predictions.size();
            for (const auto expert : experts) {
                if (residency.contains(expert)) continue;
                const auto cost = seen.contains(expert)
                                      ? kRamTransferMs
                                      : kNvmeLoadMs + kRamTransferMs;
                if (predicted.contains(expert)) {
                    predictedMissTransferMs += cost;
                    ++result.prefetchHits;
                } else {
                    ++result.prefetchMisses;
                }
            }
        }
        const auto hidden = std::min(predictedMissTransferMs, kGpuComputeMs);
        const auto stall = transferMs - hidden;
        result.hiddenTransferMs += hidden;
        result.asynchronous.totalLatencyMs += kGpuComputeMs + stall;
        result.asynchronous.stallTimeMs += stall;
        if (stall > 0.0) ++result.asynchronous.stallCount;
        result.asynchronous.peakQueueDepth =
            std::max(result.asynchronous.peakQueueDepth, misses);

        for (const auto expert : experts) {
            seen.insert(expert);
            residency.touch(expert);
        }
        previousExperts.assign(experts.begin(), experts.end());
    }
    return result;
}

std::string toJson(const Options& options, const Comparison& result) {
    const auto hiddenPercentage = result.totalTransferMs == 0.0
                                      ? 0.0
                                      : 100.0 * result.hiddenTransferMs /
                                            result.totalTransferMs;
    const auto speedup = result.asynchronous.totalLatencyMs == 0.0
                             ? 0.0
                             : result.synchronous.totalLatencyMs /
                                   result.asynchronous.totalLatencyMs;
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"tokens\": " << options.tokens << ",\n"
           << "  \"layers\": " << kLayers << ",\n"
           << "  \"top_k\": " << kTopK << ",\n"
           << "  \"simulated_gpu_compute_ms\": " << kGpuComputeMs << ",\n"
           << "  \"simulated_nvme_load_ms\": " << kNvmeLoadMs << ",\n"
           << "  \"simulated_ram_transfer_ms\": " << kRamTransferMs << ",\n"
           << "  \"synchronous\": {\n"
           << "    \"total_latency_ms\": " << result.synchronous.totalLatencyMs << ",\n"
           << "    \"stall_time_ms\": " << result.synchronous.stallTimeMs << ",\n"
           << "    \"stall_count\": " << result.synchronous.stallCount << ",\n"
           << "    \"peak_queue_depth\": " << result.synchronous.peakQueueDepth << "\n"
           << "  },\n"
           << "  \"asynchronous\": {\n"
           << "    \"total_latency_ms\": " << result.asynchronous.totalLatencyMs << ",\n"
           << "    \"stall_time_ms\": " << result.asynchronous.stallTimeMs << ",\n"
           << "    \"stall_count\": " << result.asynchronous.stallCount << ",\n"
           << "    \"peak_queue_depth\": " << result.asynchronous.peakQueueDepth << "\n"
           << "  },\n"
           << "  \"transfer_time_ms\": " << result.totalTransferMs << ",\n"
           << "  \"hidden_transfer_ms\": " << result.hiddenTransferMs << ",\n"
           << "  \"hidden_transfer_percentage\": " << hiddenPercentage << ",\n"
           << "  \"speedup\": " << speedup << ",\n"
           << "  \"prefetch_requests\": " << result.prefetchRequests << ",\n"
           << "  \"prefetch_hits\": " << result.prefetchHits << ",\n"
           << "  \"prefetch_misses\": " << result.prefetchMisses << "\n"
           << "}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        const auto result = simulate(options);
        std::ofstream report(options.report, std::ios::binary | std::ios::trunc);
        report << toJson(options, result);
        if (!report) throw std::runtime_error("failed writing pipeline report");
        const auto hidden = result.totalTransferMs == 0.0
                                ? 0.0
                                : 100.0 * result.hiddenTransferMs /
                                      result.totalTransferMs;
        std::cout << std::fixed << std::setprecision(2)
                  << "HyperMoE pipeline simulation\n"
                  << "  synchronous latency: " << result.synchronous.totalLatencyMs << " ms\n"
                  << "  asynchronous latency:" << result.asynchronous.totalLatencyMs << " ms\n"
                  << "  synchronous stalls:  " << result.synchronous.stallTimeMs << " ms\n"
                  << "  asynchronous stalls: " << result.asynchronous.stallTimeMs << " ms\n"
                  << "  transfer hidden:     " << hidden << "%\n"
                  << "  peak queue depth:    " << result.asynchronous.peakQueueDepth << '\n'
                  << "  report:              " << options.report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "pipeline benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
