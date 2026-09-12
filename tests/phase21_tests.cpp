#include "cache/HybridPolicy.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "prediction/ExpertPredictor.hpp"
#include "profiling/Profiler.hpp"
#include "runtime/metrics/RuntimeMetrics.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

int failures{};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

hypermoe::router::RouterDecision decision(hypermoe::LayerId layer,
                                           hypermoe::ExpertId expert) {
    return {layer, {expert}, {1.0F}};
}

void trainPair(hypermoe::prediction::ExpertPredictor& predictor,
               std::uint64_t stream,
               hypermoe::ExpertId source,
               hypermoe::ExpertId target) {
    predictor.observe(decision(0, source), stream);
    predictor.observe(decision(1, target), stream);
    predictor.database()->endStream(stream);
}

void testAdaptivePredictor() {
    using namespace hypermoe;
    auto database = std::make_shared<prediction::TransitionDatabase>(12, 0.80);
    prediction::AdaptivePredictionConfig configuration;
    configuration.maximumPredictions = 2;
    configuration.minimumConfidence = 0.0;
    configuration.minimumPrefetchConfidence = 0.40;
    prediction::ExpertPredictor predictor(database, configuration);

    for (std::uint64_t sample = 0; sample < 6; ++sample) {
        trainPair(predictor, sample, 2, 7);
    }
    for (std::uint64_t sample = 6; sample < 18; ++sample) {
        trainPair(predictor, sample, 2, 9);
    }
    const auto first = predictor.predictDetailed({0, {2}, {}});
    const auto second = predictor.predictDetailed({0, {2}, {}});
    expect(!first.empty() && first.front().expertId == 9 &&
               first.front().expectedLayer == 1 &&
               first.front().probability > 0.5 &&
               first.front().confidence > 0.0,
           "decayed layer-aware predictor favors recent transitions");
    expect(first.size() == second.size() &&
               std::equal(first.begin(), first.end(), second.begin(),
                          [](const auto& left, const auto& right) {
                              return left.expertId == right.expertId &&
                                     left.probability == right.probability &&
                                     left.confidence == right.confidence;
                          }),
           "adaptive prediction is deterministic");
    const auto statistics = database->predictionStatistics(
        1, std::vector<prediction::ExpertSelection>{{0, 2}});
    expect(statistics.windowLayerFrequency != 0 &&
               statistics.decayedLayerFrequency > 0.0,
           "transition database exposes windowed and decayed statistics");
    expect(predictor.shouldPrefetch(first.front()) &&
               !predictor.shouldPrefetch(
                   {first.front().expertId, first.front().probability, 0.39, 1}),
           "confidence threshold suppresses weak prefetch transfers");
}

void testPredictionQualityMath() {
    hypermoe::prediction::PredictionQualitySnapshot quality;
    quality.opportunities = 4;
    quality.predictions = 8;
    quality.top1Correct = 3;
    quality.topKCovered = 4;
    quality.usefulPredictions = 5;
    expect(std::abs(quality.top1Accuracy() - 0.75) < 1.0e-12 &&
               std::abs(quality.topKCoverage() - 1.0) < 1.0e-12 &&
               std::abs(quality.prefetchUsefulness() - 0.625) < 1.0e-12,
           "prediction quality metrics use stable denominators");
}

void testAdaptiveResidency() {
    using namespace hypermoe;
    MemoryManager memory(128, 256);
    ExpertManager manager(memory, std::make_unique<HybridPolicy>());
    manager.registerExpert({0, 0, 64, QuantizationType::Fp32, MemoryTier::Vram});
    manager.registerExpert({1, 0, 64, QuantizationType::Fp32, MemoryTier::Vram});
    manager.updatePrediction(0, 1, 1.0, 1.0);
    manager.registerExpert({2, 0, 64, QuantizationType::Fp32, MemoryTier::Nvme});
    (void)manager.requestExpert(0, 2);
    const auto zero = manager.findExpert(0, 0);
    const auto one = manager.findExpert(0, 1);
    const auto two = manager.findExpert(0, 2);
    expect(zero && one && two && zero->location != MemoryTier::Vram &&
               one->location == MemoryTier::Vram &&
               two->location == MemoryTier::Vram,
           "prediction-aware residency protects a likely expert under pressure");
    const auto residency = manager.residencySnapshot();
    const auto predicted = std::find_if(residency.begin(), residency.end(),
                                        [](const auto& item) {
                                            return item.expertId == 1;
                                        });
    expect(predicted != residency.end() && predicted->residencyScore >= 0.3 &&
               predicted->predictionProbability == 1.0,
           "residency snapshot exposes deterministic score inputs");
}

void testBatchedSynchronizationBoundary() {
    using namespace hypermoe;
    auto profiler = std::make_shared<Profiler>();
    auto backend = std::make_shared<tensor::CpuTensorBackend>(profiler);
    ExpertMlpExecutor executor(backend, tensor::activation::ActivationType::SiLU,
                               profiler);
    auto input = backend->allocateTensor({1, 2}, tensor::DType::FP32);
    auto gate = backend->allocateTensor({2, 2}, tensor::DType::FP32);
    auto up = backend->allocateTensor({2, 2}, tensor::DType::FP32);
    auto down = backend->allocateTensor({2, 2}, tensor::DType::FP32);
    auto output = backend->allocateTensor({1, 2}, tensor::DType::FP32);
    std::fill_n(static_cast<float*>(input.data()), 2, 1.0F);
    for (auto* tensor : {&gate, &up, &down}) {
        std::fill_n(static_cast<float*>(tensor->data()), 4, 0.5F);
    }
    executor.execute(input.view(), {gate.view(), up.view(), down.view()},
                     output.view());
    expect(profiler->snapshot().synchronizationCount == 0,
           "expert sub-operations do not force a synchronization barrier");
    backend->synchronizeExecution();
    expect(profiler->snapshot().synchronizationCount == 1,
           "one execution boundary records one synchronization");
}

void testMetricsInterface() {
    using namespace hypermoe;
    MemoryManager memory(256, 256);
    ExpertManager manager(memory, std::make_unique<HybridPolicy>());
    manager.registerExpert({4, 2, 64, QuantizationType::Fp32, MemoryTier::Vram});
    prediction::ExpertHistory history;
    history.record(decision(2, 4));
    prediction::PredictionQualitySnapshot quality;
    quality.opportunities = 1;
    quality.predictions = 1;
    quality.top1Correct = 1;
    quality.topKCovered = 1;
    quality.usefulPredictions = 1;
    Profiler profiler;
    profiler.recordExpertRequest(true);
    profiler.recordSynchronization(2);
    profiler.recordPrefetchUseful();
    const auto metrics = runtime::metrics::collectRuntimeMetrics(
        memory.snapshot(), history.snapshot(), quality, profiler.snapshot(),
        manager.residencySnapshot());
    expect(metrics.residentExperts == 1 && metrics.vramResidentExperts == 1 &&
               metrics.predictionTop1Accuracy == 1.0 &&
               metrics.cacheHitRate == 1.0 &&
               metrics.synchronizationCount == 2 &&
               metrics.expertFrequency.at("2:4") == 1 &&
               metrics.toJson().find("hypermoe.runtime-metrics.v1") !=
                   std::string::npos,
           "internal metrics API combines memory, MoE, and execution telemetry");
}

} // namespace

int main() {
    testAdaptivePredictor();
    testPredictionQualityMath();
    testAdaptiveResidency();
    testBatchedSynchronizationBoundary();
    testMetricsInterface();
    if (failures != 0) {
        std::cerr << failures << " Phase 21 test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 21 tests passed\n";
    return EXIT_SUCCESS;
}
