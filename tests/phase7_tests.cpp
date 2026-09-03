#include "backend/CpuBackend.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/cache_policy.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "models/metadata/JsonValue.hpp"
#include "models/qwen/QwenMoEAdapter.hpp"
#include "prediction/ExpertHistory.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/activation/Activation.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
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
                ("hypermoe-phase7-" +
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

void writeText(const std::filesystem::path& path, std::string_view contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
    if (!output) throw std::runtime_error("failed writing test manifest");
}

std::string qwenManifest() {
    return R"({
  "schema":"hypermoe.model-manifest.v1",
  "architecture":"QWEN_MOE",
  "model_name":"Synthetic Qwen MoE",
  "layer_count":1,
  "expert_count":2,
  "hidden_size":2,
  "intermediate_size":2,
  "router":{"expert_count":2,"top_k":2,"normalization":"SOFTMAX","renormalize_selected":true},
  "tensors":[
    {"name":"model.layers.0.mlp.experts.0.gate_proj.weight","shape":[2,2],"dtype":"FP32","offset":0,"size":16,"layer_id":0,"expert_id":0},
    {"name":"model.layers.0.mlp.experts.0.up_proj.weight","shape":[2,2],"dtype":"FP32","offset":16,"size":16,"layer_id":0,"expert_id":0},
    {"name":"model.layers.0.mlp.experts.0.down_proj.weight","shape":[2,2],"dtype":"FP32","offset":32,"size":16,"layer_id":0,"expert_id":0},
    {"name":"model.layers.0.mlp.experts.1.gate_proj.weight","shape":[2,2],"dtype":"FP32","offset":48,"size":16,"layer_id":0,"expert_id":1},
    {"name":"model.layers.0.mlp.experts.1.up_proj.weight","shape":[2,2],"dtype":"FP32","offset":64,"size":16,"layer_id":0,"expert_id":1},
    {"name":"model.layers.0.mlp.experts.1.down_proj.weight","shape":[2,2],"dtype":"FP32","offset":80,"size":16,"layer_id":0,"expert_id":1},
    {"name":"model.layers.0.mlp.gate.weight","shape":[2,2],"dtype":"FP32","offset":96,"size":16,"layer_id":0,"expert_id":null}
  ]
})";
}

void testMetadataParserAndAdapter() {
    using namespace hypermoe::models;
    const auto json = metadata::parseJson(
        R"({"integer":18446744073709551615,"text":"MoE \u03bc","array":[true,null]})");
    expect(json.require("integer").asUInt64() ==
               std::numeric_limits<std::uint64_t>::max() &&
               json.require("text").asString() == "MoE μ" &&
               json.require("array").asArray()[0].asBool(),
           "metadata parser preserves uint64 values, Unicode, arrays, and booleans");
    expectThrows([] { (void)metadata::parseJson(R"({"a":1,"a":2})"); },
                 "metadata parser rejects duplicate object keys");
    expectThrows([] { (void)metadata::parseJson("[1,]"); },
                 "metadata parser rejects malformed JSON");

    TemporaryDirectory directory;
    writeText(directory.path() / "metadata.json", qwenManifest());
    qwen::QwenMoEAdapter adapter;
    const auto model = adapter.loadModelMetadata(directory.path());
    const auto tensors = adapter.loadTensorIndex(directory.path());
    const auto mapping = adapter.getExpertMapping(model);
    expect(adapter.getArchitecture() == ModelArchitecture::QWEN_MOE &&
               adapter.getLayerCount(model) == 1 &&
               adapter.getExpertCount(model) == 2 && tensors.size() == 7,
           "Qwen adapter loads neutral manifest counts and tensor index");
    expect(adapter.capabilities().gatedExpertMlp &&
               adapter.capabilities().configurableTopK &&
               adapter.getRouterConfiguration(model).topK == 2,
           "adapter exposes capabilities and router configuration");
    expect(mapping.entries().size() == 2 && mapping.require(0, 0).complete() &&
               mapping.require(0, 1).gateProjection->name.find("gate_proj") !=
                   std::string::npos,
           "Qwen names map to generic complete expert projections");

    auto invalid = qwenManifest();
    const auto position = invalid.find("experts.0.gate_proj");
    invalid.replace(position, std::string("experts.0.gate_proj").size(),
                    "experts.0.unknown_proj");
    writeText(directory.path() / "invalid.json", invalid);
    expectThrows([&] { (void)adapter.loadModelMetadata(directory.path() / "invalid.json"); },
                 "Qwen adapter rejects unrecognized expert tensor names");
}

void writeFloats(hypermoe::tensor::Tensor& tensor,
                 std::initializer_list<float> values) {
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.data()));
}

void testRouterAndHistory() {
    using namespace hypermoe;
    tensor::CpuTensorBackend tensors;
    auto hidden = tensors.allocateTensor({1, 2}, tensor::DType::FP32);
    auto weights = tensors.allocateTensor({2, 3}, tensor::DType::FP32);
    writeFloats(hidden, {1.0F, 0.0F});
    writeFloats(weights, {1.0F, 3.0F, 2.0F, 0.0F, 0.0F, 0.0F});
    auto backend = std::make_shared<router::CpuRouterBackend>();
    router::Router topTwo(
        {3, 2, router::RoutingNormalization::Softmax, true}, backend);
    const auto decision = topTwo.route(4, hidden, weights);
    expect(decision.selectedExpertIds == std::vector<ExpertId>({1, 2}) &&
               std::fabs(decision.routingScores[0] + decision.routingScores[1] -
                         1.0F) < 1.0e-6F,
           "router calculates scores, selects deterministic top-k, and renormalizes");

    router::Router topOne(
        {3, 1, router::RoutingNormalization::None, false}, backend);
    expect(topOne.route(4, hidden, weights).selectedExpertIds ==
               std::vector<ExpertId>({1}),
           "router supports top-1 without assuming a fixed k");
    writeFloats(weights, {1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F});
    expect(topTwo.route(4, hidden, weights).selectedExpertIds ==
               std::vector<ExpertId>({0, 1}),
           "router breaks equal-score ties by ascending expert ID");
    expectThrows([&] { router::Router invalid({2, 3}, backend); },
                 "router rejects top-k greater than expert count");

    prediction::ExpertHistory history;
    history.record(decision);
    history.record({5, {0, 1}, {0.6F, 0.4F}});
    expect(history.frequency(4, 1) == 1 && history.frequency(5, 0) == 1 &&
               history.transitionCount({4, 1}, {5, 0}) == 1 &&
               history.previousSelections() ==
                   std::vector<prediction::ExpertSelection>({{5, 0}, {5, 1}}),
           "prediction history tracks frequency, transitions, and prior selections");
}

std::vector<std::byte> bytesFromFloats(const std::vector<float>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

void testLayerAwareExpertManager() {
    TemporaryDirectory directory;
    const std::vector<hypermoe::storage::ExpertBlob> blobs{
        {0, 0, 0, std::vector<std::byte>(4, std::byte{0x01})},
        {1, 0, 0, std::vector<std::byte>(4, std::byte{0x02})},
    };
    hypermoe::storage::ExpertStore::create(directory.path(), blobs, "{}");
    auto store = std::make_shared<hypermoe::storage::ExpertStore>(directory.path());
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
    auto compute = std::make_shared<hypermoe::backend::CpuBackend>();
    auto transfers = std::make_shared<hypermoe::TransferManager>(loader, compute, 1);
    hypermoe::MemoryManager memory(16, 16);
    hypermoe::ExpertManager manager(
        memory, std::make_unique<hypermoe::LruCachePolicy>(), transfers);
    manager.registerExpert({0, 0, 4, hypermoe::QuantizationType::Int8,
                            hypermoe::MemoryTier::Nvme});
    manager.registerExpert({0, 1, 4, hypermoe::QuantizationType::Int8,
                            hypermoe::MemoryTier::Nvme});
    (void)manager.requestExpert(0, 0);
    (void)manager.requestExpert(1, 0);
    expect(manager.findExpert(0, 0)->layer == 0 &&
               manager.findExpert(1, 0)->layer == 1,
           "ExpertManager supports identical local expert IDs in different layers");
    expectThrows([&] { (void)manager.requestExpert(0); },
                 "legacy ExpertManager lookup rejects ambiguous cross-layer IDs");
}

hypermoe::models::TensorMetadata tensorMetadata(
    std::string name,
    std::uint32_t expert,
    std::uint64_t offset) {
    return {std::move(name), hypermoe::tensor::Shape{2, 2},
            hypermoe::tensor::DType::FP32, std::nullopt, offset, 16, 0, expert};
}

void testSyntheticEndToEnd() {
    using namespace hypermoe;
    TemporaryDirectory directory;
    const std::vector<float> identity{1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> doubled{2.0F, 0.0F, 0.0F, 2.0F};
    std::vector<float> expertZero;
    expertZero.insert(expertZero.end(), identity.begin(), identity.end());
    expertZero.insert(expertZero.end(), identity.begin(), identity.end());
    expertZero.insert(expertZero.end(), identity.begin(), identity.end());
    std::vector<float> expertOne;
    expertOne.insert(expertOne.end(), identity.begin(), identity.end());
    expertOne.insert(expertOne.end(), doubled.begin(), doubled.end());
    expertOne.insert(expertOne.end(), identity.begin(), identity.end());
    const std::vector<storage::ExpertBlob> blobs{
        {0, 0, static_cast<std::uint32_t>(QuantizationType::Fp32),
         bytesFromFloats(expertZero)},
        {0, 1, static_cast<std::uint32_t>(QuantizationType::Fp32),
         bytesFromFloats(expertOne)},
    };
    storage::ExpertStore::create(directory.path(), blobs, "{\"synthetic\":true}");
    auto store = std::make_shared<storage::ExpertStore>(directory.path());
    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto compute = std::make_shared<backend::CpuBackend>();
    auto transfers = std::make_shared<TransferManager>(loader, compute, 2);
    MemoryManager memory(1024, 1024);
    ExpertManager experts(memory, std::make_unique<LruCachePolicy>(), transfers);
    auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 2);
    models::ExpertWeightMap mappings;
    for (ExpertId expert = 0; expert < 2; ++expert) {
        const auto record = store->index().find(0, expert).value();
        experts.registerExpert({expert, 0, static_cast<std::size_t>(record.size),
                                QuantizationType::Fp32, MemoryTier::Nvme});
        scheduler->registerExpert(0, expert);
        mappings.add(0, expert, models::ExpertWeightType::GATE,
                     tensorMetadata("generic.gate", expert, record.offset));
        mappings.add(0, expert, models::ExpertWeightType::UP,
                     tensorMetadata("generic.up", expert, record.offset + 16));
        mappings.add(0, expert, models::ExpertWeightType::DOWN,
                     tensorMetadata("generic.down", expert, record.offset + 32));
    }

    auto tensorBackend = std::make_shared<tensor::CpuTensorBackend>();
    auto routerBackend = std::make_shared<router::CpuRouterBackend>();
    auto router = std::make_shared<router::Router>(
        router::RouterConfig{2, 2, router::RoutingNormalization::Softmax, true},
        routerBackend);
    auto executor = std::make_shared<ExpertMlpExecutor>(tensorBackend);
    auto history = std::make_shared<prediction::ExpertHistory>();
    runtime::MoERuntime runtime(router, scheduler, experts, std::move(mappings),
                                tensorBackend, executor, history);
    auto hidden = tensorBackend->allocateTensor({1, 2}, tensor::DType::FP32);
    auto routerWeights = tensorBackend->allocateTensor({2, 2}, tensor::DType::FP32);
    writeFloats(hidden, {1.0F, 2.0F});
    writeFloats(routerWeights, {0.0F, 0.0F, 0.0F, 0.0F});
    const auto result = runtime.executeLayer(0, hidden, routerWeights);
    const auto* output = static_cast<const float*>(result.output.data());
    const auto expected0 = 1.5F * tensor::activation::silu(1.0F);
    const auto expected1 = 3.0F * tensor::activation::silu(2.0F);
    expect(result.routing.selectedExpertIds == std::vector<ExpertId>({0, 1}) &&
               std::fabs(output[0] - expected0) < 1.0e-5F &&
               std::fabs(output[1] - expected1) < 1.0e-5F,
           "synthetic MoE routes, schedules, adopts, executes, and combines experts");
    expect(experts.findExpert(0, 0)->location == MemoryTier::Vram &&
               experts.findExpert(0, 1)->location == MemoryTier::Vram &&
               scheduler->state(0, 0).state ==
                   scheduler::ExpertLifecycleState::Ready &&
               history->frequency(0, 0) == 1 && history->frequency(0, 1) == 1,
           "end-to-end execution preserves residency and prediction history state");
}

} // namespace

int main() {
    testMetadataParserAndAdapter();
    testRouterAndHistory();
    testLayerAwareExpertManager();
    testSyntheticEndToEnd();
    if (failures != 0) {
        std::cerr << failures << " Phase 7 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 7 tests passed\n";
    return EXIT_SUCCESS;
}
