#include "prediction/ExpertPredictor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

hypermoe::router::RouterDecision select(hypermoe::LayerId layer,
                                         hypermoe::ExpertId expert) {
    return {layer, {expert}, {1.0F}};
}

struct Result {
    std::uint64_t opportunities{};
    std::uint64_t predictions{};
    std::uint64_t top1{};
    std::uint64_t topK{};
    std::uint64_t useful{};
    std::chrono::nanoseconds elapsed{};
};

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path report =
            argc > 1 ? argv[1] : "prediction_report.json";
        constexpr std::size_t trainingSequences = 256;
        constexpr std::size_t measuredSequences = 4096;
        constexpr hypermoe::LayerId layers = 8;
        auto database = std::make_shared<hypermoe::prediction::TransitionDatabase>(
            128, 0.985);
        hypermoe::prediction::AdaptivePredictionConfig configuration;
        configuration.maximumPredictions = 3;
        configuration.minimumConfidence = 0.0;
        hypermoe::prediction::ExpertPredictor predictor(database, configuration);

        const auto expertFor = [](std::size_t sequence, hypermoe::LayerId layer) {
            const auto family = static_cast<hypermoe::ExpertId>(sequence % 4U);
            return static_cast<hypermoe::ExpertId>(
                (family * 7U + static_cast<std::size_t>(layer) * 3U) % 32U);
        };
        for (std::size_t sequence = 0; sequence < trainingSequences; ++sequence) {
            for (hypermoe::LayerId layer = 0; layer < layers; ++layer) {
                predictor.observe(select(layer, expertFor(sequence, layer)), sequence);
            }
            database->endStream(sequence);
        }

        Result adaptive;
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t sequence = 0; sequence < measuredSequences; ++sequence) {
            for (hypermoe::LayerId layer = 0; layer + 1U < layers; ++layer) {
                const auto current = expertFor(sequence, layer);
                const auto actual = expertFor(sequence, static_cast<hypermoe::LayerId>(layer + 1U));
                const auto predictions = predictor.predictDetailed({layer, {current}, {}});
                ++adaptive.opportunities;
                adaptive.predictions += predictions.size();
                if (!predictions.empty() && predictions.front().expertId == actual) {
                    ++adaptive.top1;
                }
                const auto matches = std::count_if(
                    predictions.begin(), predictions.end(),
                    [actual](const auto& prediction) {
                        return prediction.expertId == actual;
                    });
                if (matches != 0) ++adaptive.topK;
                adaptive.useful += static_cast<std::uint64_t>(matches);
                predictor.observe(select(layer, current),
                                  trainingSequences + sequence);
            }
            predictor.observe(
                select(static_cast<hypermoe::LayerId>(layers - 1U),
                       expertFor(sequence, static_cast<hypermoe::LayerId>(layers - 1U))),
                trainingSequences + sequence);
            database->endStream(trainingSequences + sequence);
        }
        adaptive.elapsed = std::chrono::steady_clock::now() - started;
        const auto top1 = static_cast<double>(adaptive.top1) /
                          static_cast<double>(adaptive.opportunities);
        const auto topK = static_cast<double>(adaptive.topK) /
                          static_cast<double>(adaptive.opportunities);
        const auto usefulness = static_cast<double>(adaptive.useful) /
                                static_cast<double>(adaptive.predictions);
        const auto latencyUs = std::chrono::duration<double, std::micro>(
            adaptive.elapsed).count() / static_cast<double>(adaptive.opportunities);
        std::ostringstream json;
        json << std::fixed << std::setprecision(6)
             << "{\n  \"schema\": \"hypermoe.prediction-benchmark.v1\",\n"
             << "  \"sequences\": " << measuredSequences << ",\n"
             << "  \"opportunities\": " << adaptive.opportunities << ",\n"
             << "  \"baseline\": {\"top1_accuracy\": 0.0, "
                "\"top_k_coverage\": 0.0, \"prefetch_usefulness\": 0.0},\n"
             << "  \"adaptive\": {\n"
             << "    \"top1_accuracy\": " << top1 << ",\n"
             << "    \"top_k_coverage\": " << topK << ",\n"
             << "    \"prefetch_usefulness\": " << usefulness << ",\n"
             << "    \"prediction_latency_us\": " << latencyUs << ",\n"
             << "    \"predictions\": " << adaptive.predictions << "\n"
             << "  }\n}\n";
        std::ofstream output(report, std::ios::binary | std::ios::trunc);
        output << json.str();
        if (!output) throw std::runtime_error("cannot write prediction report");
        std::cout << json.str();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "prediction benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
