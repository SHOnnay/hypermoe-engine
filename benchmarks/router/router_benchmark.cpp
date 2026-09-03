#include "prediction/ExpertHistory.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::uint32_t tokens{100000};
    std::uint32_t seed{20260903};
    std::size_t layers{32};
    std::size_t experts{64};
    std::size_t hiddenSize{16};
    std::size_t topK{2};
    std::string report{"router_report.json"};
};

template <typename T>
T parseNumber(std::string_view value, std::string_view name) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result == 0) {
        throw std::invalid_argument("invalid value for " + std::string(name));
    }
    return result;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto separator = argument.find('=');
        if (separator == std::string_view::npos) {
            throw std::invalid_argument("expected --name=value");
        }
        const auto name = argument.substr(0, separator);
        const auto value = argument.substr(separator + 1);
        if (name == "--tokens") options.tokens = parseNumber<std::uint32_t>(value, name);
        else if (name == "--seed") options.seed = parseNumber<std::uint32_t>(value, name);
        else if (name == "--layers") options.layers = parseNumber<std::size_t>(value, name);
        else if (name == "--experts") options.experts = parseNumber<std::size_t>(value, name);
        else if (name == "--hidden") options.hiddenSize = parseNumber<std::size_t>(value, name);
        else if (name == "--top-k") options.topK = parseNumber<std::size_t>(value, name);
        else if (name == "--report") options.report = value;
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    return options;
}

struct Results {
    std::vector<std::uint64_t> frequency;
    std::vector<double> averageScoreByRank;
    std::uint64_t repeatedSelections{};
    std::uint64_t localityComparisons{};
    std::uint64_t recordedTransitions{};
    double elapsedMs{};
    double decisionsPerSecond{};
};

Results run(const Options& options) {
    if (options.layers > std::numeric_limits<hypermoe::LayerId>::max()) {
        throw std::invalid_argument("layer count exceeds runtime IDs");
    }
    hypermoe::router::RouterConfig config{
        options.experts, options.topK,
        hypermoe::router::RoutingNormalization::Softmax, true};
    config.validate();
    auto routerBackend = std::make_shared<hypermoe::router::CpuRouterBackend>();
    hypermoe::router::Router router(config, routerBackend);
    hypermoe::tensor::CpuTensorBackend tensors;
    auto hidden = tensors.allocateTensor({1, options.hiddenSize},
                                         hypermoe::tensor::DType::FP32);
    auto weights = tensors.allocateTensor({options.hiddenSize, options.experts},
                                          hypermoe::tensor::DType::FP32);
    auto* hiddenValues = static_cast<float*>(hidden.data());
    auto* weightValues = static_cast<float*>(weights.data());
    std::mt19937 generator(options.seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (std::size_t index = 0; index < weights.shape().elementCount(); ++index) {
        weightValues[index] = normal(generator) * 0.25F;
    }
    std::fill(hiddenValues, hiddenValues + options.hiddenSize, 0.0F);

    Results result;
    result.frequency.resize(options.experts);
    result.averageScoreByRank.resize(options.topK);
    std::vector<hypermoe::ExpertId> previous;
    hypermoe::prediction::ExpertHistory history;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t token = 0; token < options.tokens; ++token) {
        for (std::size_t index = 0; index < options.hiddenSize; ++index) {
            hiddenValues[index] = 0.92F * hiddenValues[index] + 0.08F * normal(generator);
        }
        const auto layer = static_cast<hypermoe::LayerId>(token % options.layers);
        const auto decision = router.route(layer, hidden, weights);
        history.record(decision);
        for (std::size_t rank = 0; rank < decision.selectedExpertIds.size(); ++rank) {
            ++result.frequency[decision.selectedExpertIds[rank]];
            result.averageScoreByRank[rank] += decision.routingScores[rank];
        }
        if (!previous.empty()) {
            for (const auto expert : decision.selectedExpertIds) {
                if (std::find(previous.begin(), previous.end(), expert) != previous.end()) {
                    ++result.repeatedSelections;
                }
                ++result.localityComparisons;
            }
        }
        previous = decision.selectedExpertIds;
    }
    result.elapsedMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();
    result.decisionsPerSecond =
        static_cast<double>(options.tokens) / (result.elapsedMs / 1000.0);
    for (auto& score : result.averageScoreByRank) score /= options.tokens;
    result.recordedTransitions = history.snapshot().layerTransitions.size();
    return result;
}

std::string toJson(const Options& options, const Results& result) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n  \"tokens\": " << options.tokens << ",\n"
           << "  \"seed\": " << options.seed << ",\n"
           << "  \"layers\": " << options.layers << ",\n"
           << "  \"experts\": " << options.experts << ",\n"
           << "  \"top_k\": " << options.topK << ",\n"
           << "  \"elapsed_ms\": " << result.elapsedMs << ",\n"
           << "  \"decisions_per_second\": " << result.decisionsPerSecond << ",\n"
           << "  \"expert_locality\": "
           << (result.localityComparisons == 0
                   ? 0.0
                   : static_cast<double>(result.repeatedSelections) /
                         static_cast<double>(result.localityComparisons))
           << ",\n  \"unique_layer_transitions\": "
           << result.recordedTransitions << ",\n  \"expert_frequency\": [";
    for (std::size_t index = 0; index < result.frequency.size(); ++index) {
        if (index != 0) output << ',';
        output << result.frequency[index];
    }
    output << "],\n  \"routing_distribution\": [";
    const auto selections = static_cast<double>(options.tokens) *
                            static_cast<double>(options.topK);
    for (std::size_t index = 0; index < result.frequency.size(); ++index) {
        if (index != 0) output << ',';
        output << static_cast<double>(result.frequency[index]) / selections;
    }
    output << "],\n  \"average_score_by_rank\": [";
    for (std::size_t index = 0; index < result.averageScoreByRank.size(); ++index) {
        if (index != 0) output << ',';
        output << result.averageScoreByRank[index];
    }
    output << "]\n}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        const auto results = run(options);
        std::ofstream report(options.report, std::ios::binary | std::ios::trunc);
        report << toJson(options, results);
        if (!report) throw std::runtime_error("failed writing router benchmark report");
        const auto locality = results.localityComparisons == 0
                                  ? 0.0
                                  : static_cast<double>(results.repeatedSelections) /
                                        static_cast<double>(results.localityComparisons);
        std::cout << std::fixed << std::setprecision(3)
                  << "HyperMoE router benchmark\n"
                  << "  Tokens:       " << options.tokens << '\n'
                  << "  Top-k:        " << options.topK << '\n'
                  << "  Decisions/s:  " << results.decisionsPerSecond << '\n'
                  << "  Locality:     " << locality * 100.0 << "%\n"
                  << "  Report:       " << options.report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "router benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
