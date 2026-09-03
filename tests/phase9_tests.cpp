#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tools/model_convert/ExpertPacker.hpp"
#include "tools/model_convert/WeightConverter.hpp"
#include "validation/CorrectnessOracle.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
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
                ("hypermoe-phase9-" +
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

std::vector<std::byte> floatBytes(std::span<const float> values) {
    std::vector<std::byte> bytes(values.size_bytes());
    std::memcpy(bytes.data(), values.data(), values.size_bytes());
    return bytes;
}

void appendFloats(std::vector<std::byte>& bytes, std::initializer_list<float> values) {
    const auto offset = bytes.size();
    bytes.resize(offset + values.size() * sizeof(float));
    std::memcpy(bytes.data() + offset, values.begin(), values.size() * sizeof(float));
}

void writeArtifact(const std::filesystem::path& root) {
    {
        std::ofstream config(root / "config.json", std::ios::binary);
        config << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Phase9-Fixture","num_hidden_layers":1,"num_experts":2,"hidden_size":2,"moe_intermediate_size":3,"num_experts_per_tok":1,"norm_topk_prob":true})";
    }
    std::vector<std::byte> payload;
    // Qwen3 fused gate/up is [expert, 2 * intermediate, hidden], OUTPUT_INPUT.
    appendFloats(payload, {0.5F, -0.25F, 1.0F, 0.5F, -0.5F, 1.0F,
                           1.0F, 0.0F, 0.0F, 1.0F, 0.5F, 0.5F});
    appendFloats(payload, {-0.5F, 0.25F, 0.75F, -1.0F, 1.0F, 0.5F,
                           0.25F, 1.0F, 1.0F, -0.5F, 0.5F, 0.25F});
    // Fused down is [expert, hidden, intermediate], OUTPUT_INPUT.
    appendFloats(payload, {1.0F, 0.5F, -0.25F, 0.0F, 1.0F, 0.5F});
    appendFloats(payload, {0.25F, 1.0F, 0.5F, -0.5F, 0.5F, 1.0F});
    // Router is [expert, hidden], OUTPUT_INPUT.
    appendFloats(payload, {2.0F, -0.5F, -1.0F, 1.5F});
    std::string header = R"({"model.layers.0.mlp.experts.gate_up_proj":{"dtype":"F32","shape":[2,6,2],"data_offsets":[0,96]},"model.layers.0.mlp.experts.down_proj":{"dtype":"F32","shape":[2,2,3],"data_offsets":[96,144]},"model.layers.0.mlp.gate.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[144,160]},"__metadata__":{"format":"pt"}})";
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    const auto headerSize = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((headerSize >> shift) & 0xffU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    if (!output) throw std::runtime_error("failed writing Phase 9 artifact");
}

hypermoe::models::ExpertWeightMap makeWeightMap(
    const hypermoe::models::ModelManifest& manifest,
    std::uint32_t layer,
    std::uint32_t expert) {
    using namespace hypermoe;
    const auto* mapping = manifest.findExpert(layer, expert);
    if (!mapping) throw std::runtime_error("packed mapping missing");
    models::ExpertWeightMap result;
    const auto add = [&](models::ExpertWeightType type,
                         const models::ProjectionLocation& projection) {
        const auto* tensor = manifest.findTensor(projection.tensorName);
        if (!tensor) throw std::runtime_error("packed tensor missing");
        result.add(layer, expert, type,
                   {tensor->name, projection.shape, tensor->dtype, std::nullopt,
                    projection.offset, projection.size, layer, expert});
    };
    add(models::ExpertWeightType::GATE, mapping->gate);
    add(models::ExpertWeightType::UP, mapping->up);
    add(models::ExpertWeightType::DOWN, mapping->down);
    return result;
}

void testWeightConverter() {
    using namespace hypermoe;
    const std::vector<float> source{1, 2, 3, 4, 5, 6};
    const auto converted = conversion::WeightConverter::convert(
        floatBytes(source), {2, 3}, tensor::DType::FP32,
        models::TensorLayout::OutputInput);
    const std::vector<float> expected{1, 4, 2, 5, 3, 6};
    expect(converted.shape == tensor::Shape{3, 2} &&
               converted.layout == models::TensorLayout::InputOutput &&
               converted.dtype == tensor::DType::FP32 &&
               std::memcmp(converted.bytes.data(), expected.data(),
                           converted.bytes.size()) == 0,
           "weight converter transposes OUTPUT_INPUT while preserving dtype bytes");
    expectThrows([&] {
        (void)conversion::WeightConverter::convert(
            std::span<const std::byte>(converted.bytes).first(4), {2, 3},
            tensor::DType::FP32, models::TensorLayout::OutputInput);
    }, "weight converter rejects inconsistent storage metadata");
}

void testRealArtifactPipeline() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto artifact = temporary.path() / "artifact";
    const auto packedPath = temporary.path() / "hypermoe_model";
    std::filesystem::create_directories(artifact);
    writeArtifact(artifact);

    importer::qwen::QwenImporter importer;
    const auto sourceManifest = importer.inspect(artifact);
    const auto report = conversion::ExpertPacker{}.pack(
        sourceManifest, artifact, packedPath);
    expect(report.experts == 2 && report.projections == 6 &&
               std::filesystem::exists(packedPath / "manifest.json") &&
               std::filesystem::exists(packedPath / "experts.bin") &&
               std::filesystem::exists(packedPath / "experts.index"),
           "Qwen SafeTensors artifact packs into the complete HyperMoE model format");

    const auto packedManifest = models::ModelManifest::load(packedPath / "manifest.json");
    auto store = std::make_shared<storage::ExpertStore>(packedPath);
    const auto record = store->index().find(0, 0);
    const auto gateRecord = store->index().findProjection(
        0, 0, storage::ProjectionType::Gate);
    expect(record && gateRecord && gateRecord->shape[0] == 2 &&
               gateRecord->shape[1] == 3 && store->index().projections().size() == 6 &&
               store->mappedProjection(0, 0, storage::ProjectionType::Gate).size() == 24,
           "projection-aware index provides checksummed random access inside expert payloads");
    if (!record) return;

    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto compute = std::make_shared<backend::CpuBackend>();
    auto transfers = std::make_shared<TransferManager>(loader, compute, 1);
    auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 1);
    scheduler->registerExpert(0, 0);
    scheduler::ScheduleRequest request;
    request.layerId = 0;
    request.expertId = 0;
    const auto loaded = scheduler->schedule(request).future().get();
    expect(loaded.success && loaded.transfer.deviceBuffer,
           "scheduler loads one packed expert through store and transfer manager");

    MemoryManager memory(1U << 20U, 1U << 20U);
    ExpertManager manager(memory, std::make_unique<LruCachePolicy>());
    manager.registerExpert({0, 0, static_cast<std::size_t>(record->size),
                            QuantizationType::Fp32, MemoryTier::Nvme});
    manager.adoptDeviceWeights(0, 0, loaded.transfer.deviceBuffer);
    scheduler->acquire(0, 0);
    const auto weightMap = makeWeightMap(packedManifest, 0, 0);
    auto tensorBackend = std::make_shared<tensor::CpuTensorBackend>();
    auto input = tensorBackend->allocateTensor({1, 2}, tensor::DType::FP32);
    auto output = tensorBackend->allocateTensor({1, 2}, tensor::DType::FP32);
    const std::vector<float> hidden{1.0F, 0.5F};
    std::memcpy(input.data(), hidden.data(), hidden.size() * sizeof(float));
    {
        auto lease = manager.acquireResidentExpert(0, 0);
        const auto payload = lease.view({static_cast<std::size_t>(record->size)},
                                        tensor::DType::INT8);
        const auto views = weightMap.createViews(0, 0, payload, record->offset);
        ExpertMlpExecutor executor(tensorBackend);
        executor.execute(input, views, output);
        const std::vector<float> gate{0.5F, 1.0F, -0.5F, -0.25F, 0.5F, 1.0F};
        const std::vector<float> up{1.0F, 0.0F, 0.5F, 0.0F, 1.0F, 0.5F};
        const std::vector<float> down{1.0F, 0.0F, 0.5F, 1.0F, -0.25F, 0.5F};
        const auto expected = validation::CorrectnessOracle::expertMlp(
            hidden, 1, 2, gate, up, down, 3);
        const auto actual = std::span<const float>(
            static_cast<const float*>(output.data()), 2);
        expect(validation::CorrectnessOracle::compare(
                   actual, expected,
                   validation::CorrectnessOracle::toleranceFor(
                       tensor::DType::FP32)).matches,
               "packed expert CPU forward agrees with independent scalar oracle");
        expectThrows([&] { manager.moveExpert(0, 0, MemoryTier::Nvme); },
                     "active TensorView residency lease prevents premature eviction");
    }
    scheduler->release(0, 0);
    manager.moveExpert(0, 0, MemoryTier::Nvme);
    expect(manager.findExpert(0, 0)->location == MemoryTier::Nvme,
           "expert becomes evictable after executor and scheduler release ownership");

    const auto* routerTensor = packedManifest.findTensor(
        packedManifest.router.tensors.front().tensorName);
    if (!routerTensor) throw std::runtime_error("packed router missing");
    auto routerWeights = tensorBackend->allocateTensor(routerTensor->shape,
                                                        tensor::DType::FP32);
    std::ifstream packedData(packedPath / "experts.bin", std::ios::binary);
    packedData.seekg(static_cast<std::streamoff>(routerTensor->offset));
    packedData.read(static_cast<char*>(routerWeights.data()),
                    static_cast<std::streamsize>(routerTensor->size));
    auto routerBackend = std::make_shared<router::CpuRouterBackend>();
    router::Router runtimeRouter(packedManifest.router.config, routerBackend);
    const auto routed = runtimeRouter.route(0, input, routerWeights);
    const std::vector<float> routerIo{2.0F, -1.0F, -0.5F, 1.5F};
    const auto oracleRoute = validation::CorrectnessOracle::route(
        0, hidden, routerIo, packedManifest.router.config);
    expect(routed.selectedExpertIds == oracleRoute.selectedExpertIds &&
               validation::CorrectnessOracle::compare(
                   routed.routingScores, oracleRoute.routingScores,
                   validation::CorrectnessOracle::toleranceFor(tensor::DType::FP32)).matches,
           "router top-k identities and normalized scores agree with oracle");
}

void testTolerancePolicy() {
    using hypermoe::tensor::DType;
    using hypermoe::validation::CorrectnessOracle;
    expect(CorrectnessOracle::toleranceFor(DType::FP32).absolute == 1.0e-5F &&
               CorrectnessOracle::toleranceFor(DType::FP16).absolute > 1.0e-5F &&
               CorrectnessOracle::toleranceFor(DType::BF16).absolute >=
                   CorrectnessOracle::toleranceFor(DType::FP16).absolute &&
               CorrectnessOracle::toleranceFor(DType::INT8).absolute >
                   CorrectnessOracle::toleranceFor(DType::BF16).absolute,
           "correctness oracle exposes explicit FP32/FP16/BF16/INT8 tolerances");
}

void testPackedProjectionCorruption() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto artifact = temporary.path() / "artifact";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(artifact);
    writeArtifact(artifact);
    importer::qwen::QwenImporter importer;
    (void)conversion::ExpertPacker{}.pack(importer.inspect(artifact), artifact, packed);
    const auto index = storage::ExpertIndex::load(packed / "experts.index");
    const auto gate = index.findProjection(0, 0, storage::ProjectionType::Gate);
    if (!gate) throw std::runtime_error("fixture projection index missing");
    {
        std::fstream data(packed / "experts.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        data.seekp(static_cast<std::streamoff>(gate->offset));
        data.put(static_cast<char>(0x7f));
    }
    storage::ExpertStore store(packed);
    expectThrows(
        [&] { (void)store.mappedProjection(0, 0, storage::ProjectionType::Gate); },
        "packed projection checksum detects artifact corruption");
}

} // namespace

int main() {
    testWeightConverter();
    testRealArtifactPipeline();
    testTolerancePolicy();
    testPackedProjectionCorruption();
    if (failures != 0) {
        std::cerr << failures << " Phase 9 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 9 tests passed\n";
    return EXIT_SUCCESS;
}
