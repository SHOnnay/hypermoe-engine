#include "prediction/ExpertPredictor.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
    std::size_t tokens{100000};
    std::filesystem::path report{"end_to_end_report.json"};
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument.starts_with("--tokens=")) {
            result.tokens = std::stoull(argument.substr(9));
        } else if (argument.starts_with("--report=")) {
            result.report = argument.substr(9);
        } else {
            throw std::invalid_argument("unknown benchmark option: " + argument);
        }
    }
    if (result.tokens == 0) throw std::invalid_argument("token count must be nonzero");
    return result;
}

std::uint64_t identity(std::uint32_t layer, std::uint32_t expert) noexcept {
    return (static_cast<std::uint64_t>(layer) << 32U) | expert;
}

} // namespace

int main(int argc, char** argv) {
    try {
        using namespace hypermoe;
        constexpr std::size_t batchSize = 64;
        constexpr std::size_t layers = 8;
        constexpr std::size_t expertCount = 32;
        constexpr std::size_t hiddenSize = 16;
        constexpr std::size_t topK = 2;
        constexpr std::size_t cacheCapacity = 96;
        constexpr double modeledLoadMs = 0.22;
        const auto settings = options(argc, argv);

        tensor::CpuTensorBackend tensors;
        auto weights = tensors.allocateTensor({hiddenSize, expertCount},
                                              tensor::DType::FP32);
        auto* weightValues = static_cast<float*>(weights.data());
        for (std::size_t hidden = 0; hidden < hiddenSize; ++hidden) {
            for (std::size_t expert = 0; expert < expertCount; ++expert) {
                weightValues[hidden * expertCount + expert] = static_cast<float>(
                    std::sin(static_cast<double>((hidden + 1) * (expert + 3))) +
                    0.25 * std::cos(static_cast<double>(expert + hidden * 7)));
            }
        }
        router::Router router(
            {expertCount, topK, router::RoutingNormalization::Softmax, true},
            std::make_shared<router::CpuRouterBackend>());
        auto database = std::make_shared<prediction::TransitionDatabase>(64);
        prediction::ExpertPredictor predictor(database, topK, 0.01);

        std::unordered_map<std::uint64_t, std::uint64_t> cache;
        std::uint64_t clock{};
        std::uint64_t expertRequests{};
        std::uint64_t cacheHits{};
        std::uint64_t predicted{};
        std::uint64_t predictionHits{};
        std::uint64_t misses{};
        double checksum{};
        std::chrono::nanoseconds routingTime{};
        std::chrono::nanoseconds executionTime{};

        for (std::size_t base = 0; base < settings.tokens; base += batchSize) {
            const auto count = std::min(batchSize, settings.tokens - base);
            std::vector<std::vector<ExpertId>> previous(count);
            for (std::size_t layer = 0; layer < layers; ++layer) {
                auto hidden = tensors.allocateTensor({count, hiddenSize},
                                                     tensor::DType::FP32);
                auto* values = static_cast<float*>(hidden.data());
                for (std::size_t token = 0; token < count; ++token) {
                    const auto preferred = static_cast<std::size_t>(
                        (base + token) * 17U + layer * 11U) % expertCount;
                    for (std::size_t axis = 0; axis < hiddenSize; ++axis) {
                        values[token * hiddenSize + axis] =
                            weightValues[axis * expertCount + preferred] +
                            static_cast<float>((token + axis + layer) % 5U) * 0.002F;
                    }
                }
                const auto routeStart = std::chrono::steady_clock::now();
                const auto batch = router.routeBatch(static_cast<LayerId>(layer),
                                                     hidden, weights);
                routingTime += std::chrono::steady_clock::now() - routeStart;
                const auto executionStart = std::chrono::steady_clock::now();
                for (std::size_t token = 0; token < count; ++token) {
                    std::vector<scheduler::PredictedExpertRequest> predictions;
                    if (layer != 0) {
                        predictions = predictor.predict(
                            {static_cast<LayerId>(layer - 1), previous[token], {}});
                        predicted += predictions.size();
                    }
                    for (const auto& prediction : predictions) {
                        cache[identity(prediction.layerId, prediction.expertId)] = ++clock;
                    }
                    const auto& decision = batch.tokens[token];
                    for (const auto expert : decision.selectedExpertIds) {
                        ++expertRequests;
                        const auto key = identity(static_cast<std::uint32_t>(layer), expert);
                        const auto hit = cache.find(key);
                        if (hit != cache.end()) {
                            ++cacheHits;
                            hit->second = ++clock;
                        } else {
                            ++misses;
                            cache[key] = ++clock;
                        }
                        if (std::any_of(predictions.begin(), predictions.end(),
                                        [&](const auto& prediction) {
                                            return prediction.expertId == expert;
                                        })) {
                            ++predictionHits;
                        }
                    }
                    predictor.observe(decision, base + token);
                    if (layer + 1 == layers) database->endStream(base + token);
                    previous[token] = decision.selectedExpertIds;
                    for (std::size_t rank = 0; rank < decision.routingScores.size(); ++rank) {
                        checksum += decision.routingScores[rank] *
                                    static_cast<double>(decision.selectedExpertIds[rank] + 1U);
                    }
                    while (cache.size() > cacheCapacity) {
                        const auto victim = std::min_element(
                            cache.begin(), cache.end(), [](const auto& left, const auto& right) {
                                return left.second < right.second;
                            });
                        cache.erase(victim);
                    }
                }
                executionTime += std::chrono::steady_clock::now() - executionStart;
            }
        }
        const auto routingMs =
            std::chrono::duration<double, std::milli>(routingTime).count();
        const auto executionMs =
            std::chrono::duration<double, std::milli>(executionTime).count();
        const auto loadingMs = static_cast<double>(misses) * modeledLoadMs;
        const auto cacheHitRate = expertRequests == 0
            ? 0.0 : static_cast<double>(cacheHits) / static_cast<double>(expertRequests);
        const auto prefetchAccuracy = predicted == 0
            ? 0.0 : static_cast<double>(predictionHits) / static_cast<double>(predicted);
        std::ofstream report(settings.report, std::ios::binary | std::ios::trunc);
        if (!report) throw std::runtime_error("cannot create end-to-end report");
        report << std::fixed << std::setprecision(6)
               << "{\n  \"tokens\": " << settings.tokens
               << ",\n  \"layers\": " << layers
               << ",\n  \"batch_size\": " << batchSize
               << ",\n  \"top_k\": " << topK
               << ",\n  \"routing_latency_ms\": " << routingMs
               << ",\n  \"routing_tokens_per_second\": "
               << static_cast<double>(settings.tokens * layers) / (routingMs / 1000.0)
               << ",\n  \"expert_loading_time_modeled_ms\": " << loadingMs
               << ",\n  \"prefetch_accuracy\": " << prefetchAccuracy
               << ",\n  \"cache_hit_rate\": " << cacheHitRate
               << ",\n  \"execution_time_ms\": " << executionMs
               << ",\n  \"expert_requests\": " << expertRequests
               << ",\n  \"cache_misses\": " << misses
               << ",\n  \"checksum\": " << checksum << "\n}\n";
        if (!report) throw std::runtime_error("failed writing end-to-end report");
        std::cout << std::fixed << std::setprecision(3)
                  << "HyperMoE Phase 8 end-to-end benchmark\n"
                  << "  Tokens:            " << settings.tokens << '\n'
                  << "  Routing tokens/s:  "
                  << static_cast<double>(settings.tokens * layers) /
                         (routingMs / 1000.0) << '\n'
                  << "  Prefetch accuracy: " << prefetchAccuracy * 100.0 << "%\n"
                  << "  Cache hit rate:    " << cacheHitRate * 100.0 << "%\n"
                  << "  Report:            " << settings.report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "end-to-end benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
