#include "backend/CpuBackend.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "models/ModelManifest.hpp"
#include "prediction/ExpertPredictor.hpp"
#include "profiling/Profiler.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void expectThrows(Function&& function, std::string_view message) {
    try {
        function();
        expect(false, message);
    } catch (const std::exception&) {
        expect(true, message);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-phase8-" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void writeText(const std::filesystem::path& path, std::string_view value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
    if (!output) throw std::runtime_error("failed writing test text file");
}

void writeSafeTensors(const std::filesystem::path& path) {
    std::string header = R"({"model.layers.0.mlp.experts.gate_up_proj":{"dtype":"F32","shape":[2,4,2],"data_offsets":[0,64]},"model.layers.0.mlp.experts.down_proj":{"dtype":"F32","shape":[2,2,2],"data_offsets":[64,96]},"model.layers.0.mlp.gate.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[96,112]},"__metadata__":{"format":"pt"}})";
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((size >> shift) & 0xFFU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::vector<char> data(112, '\0');
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!output) throw std::runtime_error("failed writing SafeTensors fixture");
}

void writeQwenArtifact(const std::filesystem::path& path) {
    writeText(path / "config.json", R"({
      "architectures":["Qwen3MoeForCausalLM"],
      "model_type":"qwen3_moe",
      "_name_or_path":"Qwen/Test-MoE",
      "num_hidden_layers":1,
      "num_experts":2,
      "hidden_size":2,
      "moe_intermediate_size":2,
      "num_experts_per_tok":1,
      "norm_topk_prob":true
    })");
    writeSafeTensors(path / "model.safetensors");
}

void testImporterAndManifest() {
    TemporaryDirectory directory;
    writeQwenArtifact(directory.path());
    hypermoe::importer::qwen::QwenImporter importer;
    expect(importer.canImport(directory.path()), "Qwen importer recognizes artifact directory");
    const auto manifest = importer.inspect(directory.path());
    expect(manifest.architecture == hypermoe::models::ModelArchitecture::QWEN_MOE &&
               manifest.sourceArchitecture == "Qwen3MoeForCausalLM" &&
               manifest.tensors.size() == 3 && manifest.experts.size() == 2 &&
               manifest.router.tensors.size() == 1 &&
               manifest.router.tensors.front().layerId == 0,
           "Qwen importer discovers real SafeTensors metadata without payload loading");
    const auto* expertZero = manifest.findExpert(0, 0);
    const auto* gateUp = manifest.findTensor(
        "model.layers.0.mlp.experts.gate_up_proj");
    expect(expertZero != nullptr && gateUp != nullptr &&
               expertZero->gate.offset == gateUp->offset &&
               expertZero->up.offset == gateUp->offset + 16 &&
               expertZero->down.layout == hypermoe::models::TensorLayout::OutputInput,
           "Qwen fused tensors map to validated per-expert projection slices");

    const auto manifestPath = directory.path() / "hypermoe-manifest.json";
    (void)importer.importModel(directory.path(), manifestPath);
    const auto loaded = hypermoe::models::ModelManifest::load(manifestPath);
    expect(loaded.modelName == "Qwen/Test-MoE" && loaded.experts.size() == 2 &&
               loaded.findExpert(0, 1) != nullptr,
           "HyperMoE manifest survives validated JSON round-trip");

    auto invalid = loaded;
    invalid.experts.front().gate.tensorName = "missing.tensor";
    expectThrows([&] { invalid.validate(); },
                 "manifest rejects expert mappings to unknown tensors");
    std::filesystem::resize_file(directory.path() / "model.safetensors", 16);
    expectThrows([&] { (void)importer.inspect(directory.path()); },
                 "SafeTensors importer rejects truncated tensor payload ranges");
    writeText(directory.path() / "config.json", R"({"model_type":"dense"})");
    expectThrows([&] { (void)importer.inspect(directory.path()); },
                 "Qwen importer rejects non-Qwen architecture metadata");
}

void writeFloats(hypermoe::tensor::Tensor& tensor,
                 std::initializer_list<float> values) {
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.data()));
}

void testBatchRouter() {
    using namespace hypermoe;
    tensor::CpuTensorBackend tensors;
    auto hidden = tensors.allocateTensor({3, 2}, tensor::DType::FP32);
    auto weights = tensors.allocateTensor({2, 3}, tensor::DType::FP32);
    writeFloats(hidden, {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F});
    writeFloats(weights, {3.0F, 2.0F, 1.0F, 0.0F, 4.0F, 1.0F});
    auto backend = std::make_shared<router::CpuRouterBackend>();
    router::Router router({3, 2, router::RoutingNormalization::Softmax, true},
                          backend);
    const auto batch = router.routeBatch(7, hidden, weights);
    expect(batch.valid() && batch.tokens.size() == 3 &&
               batch.tokens[0].selectedExpertIds == std::vector<ExpertId>({0, 1}) &&
               batch.tokens[1].selectedExpertIds == std::vector<ExpertId>({1, 2}) &&
               batch.expertGroups.size() == 3,
           "batch router selects top-k per token and groups tokens by expert");
    expectThrows([&] { (void)router.route(7, hidden, weights); },
                 "single-token router rejects a batch");
}

void testPredictionDatabase() {
    using namespace hypermoe;
    auto database = std::make_shared<prediction::TransitionDatabase>(8);
    prediction::ExpertPredictor predictor(database, 2, 0.0);
    for (int sample = 0; sample < 20; ++sample) {
        predictor.observe({0, {1, 2}, {0.6F, 0.4F}});
        predictor.observe({1, {3, 4}, {0.7F, 0.3F}});
    }
    const auto predictions = predictor.predict({0, {1, 2}, {}});
    expect(predictions.size() == 2 && predictions[0].layerId == 1 &&
               predictions[0].expertId == 3 && predictions[1].expertId == 4,
           "statistical predictor ranks learned layer transitions deterministically");
    expect(database->transitionCount({0, 1}, {1, 3}) == 20 &&
               database->cooccurrenceCount({1, 3}, {1, 4}) == 20 &&
               database->snapshot().recentSelections.size() == 8,
           "transition database tracks frequency, transitions, co-occurrence, and recency");

    prediction::TransitionDatabase streams;
    streams.record({0, {1}, {1.0F}}, 10);
    streams.record({0, {2}, {1.0F}}, 11);
    streams.record({1, {3}, {1.0F}}, 10);
    expect(streams.transitionCount({0, 1}, {1, 3}) == 1 &&
               streams.transitionCount({0, 2}, {1, 3}) == 0,
           "transition database isolates independent inference streams");
    streams.endStream(10);
    expect(streams.previousSelections(10).empty(),
           "completed inference streams release predecessor state");
}

void testPredictivePrefetchIntegration() {
    using namespace hypermoe;
    TemporaryDirectory directory;
    const std::vector<storage::ExpertBlob> blobs{
        {1, 3, 0, std::vector<std::byte>(16, std::byte{0x2A})},
    };
    storage::ExpertStore::create(directory.path(), blobs, "{}");
    auto store = std::make_shared<storage::ExpertStore>(directory.path());
    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto transfers = std::make_shared<TransferManager>(
        loader, std::make_shared<backend::CpuBackend>(), 1);
    auto profiler = std::make_shared<Profiler>();
    scheduler::Scheduler scheduler(transfers, profiler, 1);
    scheduler.registerExpert(1, 3);
    auto database = std::make_shared<prediction::TransitionDatabase>();
    prediction::ExpertPredictor predictor(database, 1, 0.0);
    for (int sample = 0; sample < 5; ++sample) {
        predictor.observe({0, {1}, {1.0F}});
        predictor.observe({1, {3}, {1.0F}});
    }
    prediction::ExpertHistory history;
    auto handles = predictor.observeAndPrefetch({0, {1}, {1.0F}}, history, scheduler);
    expect(handles.size() == 1 && handles.front().future().get().success,
           "router observation submits predicted expert through scheduler prefetch");
    scheduler::ScheduleRequest active;
    active.layerId = 1;
    active.expertId = 3;
    const auto result = scheduler.schedule(active).future().get();
    expect(result.success && result.transfer.deviceBuffer &&
               result.transfer.deviceBuffer->size() == 16 &&
               profiler->snapshot().prefetchHits == 1 && history.frequency(0, 1) == 1,
           "active request reuses retained prefetched buffer and records a hit");
}

} // namespace

int main() {
    testImporterAndManifest();
    testBatchRouter();
    testPredictionDatabase();
    testPredictivePrefetchIntegration();
    if (failures != 0) {
        std::cerr << failures << " Phase 8 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 8 tests passed\n";
    return EXIT_SUCCESS;
}
