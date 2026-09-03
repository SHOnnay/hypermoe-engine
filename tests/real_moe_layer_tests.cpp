#include "backend/CpuBackend.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "importer/SafeTensorShardManager.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "importer/validation/CheckpointValidator.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/precision/DTypeConverter.hpp"
#include "tensor/quantization/QuantizationPolicy.hpp"
#include "transformer/MoELayer.hpp"
#include "tools/model_convert/ExpertPacker.hpp"
#include "validation/CorrectnessOracle.hpp"

#include <atomic>
#include <bit>
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
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template <typename Function>
void expectThrows(Function&& function, std::string_view message) {
    try { function(); expect(false, message); }
    catch (const std::exception&) { expect(true, message); }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
            ("hypermoe-phase10-" + std::to_string(sequence.fetch_add(1)));
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

std::vector<std::byte> bf16Bytes(std::initializer_list<float> values) {
    std::vector<std::byte> result;
    result.reserve(values.size() * 2);
    for (const auto value : values) {
        const auto bits = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(value) >> 16U);
        result.push_back(static_cast<std::byte>(bits & 0xffU));
        result.push_back(static_cast<std::byte>(bits >> 8U));
    }
    return result;
}

void writeSafeTensor(const std::filesystem::path& path,
                     std::string header,
                     std::span<const std::byte> payload) {
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(path, std::ios::binary);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((size >> shift) & 0xffU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    if (!output) throw std::runtime_error("failed writing SafeTensors shard");
}

void writeShardedQwen(const std::filesystem::path& root) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Phase10-Sharded-BF16","num_hidden_layers":1,"num_experts":2,"hidden_size":2,"moe_intermediate_size":2,"num_experts_per_tok":1,"norm_topk_prob":true})";
    const auto gateUp = bf16Bytes({
        1, 0, 0, 1, 1, 0, 0, 1,
        0.5F, 0, 0, 0.5F, 1, 0, 0, 1});
    writeSafeTensor(root / "model-00001-of-00002.safetensors",
        R"({"model.layers.0.mlp.experts.gate_up_proj":{"dtype":"BF16","shape":[2,4,2],"data_offsets":[0,32]},"__metadata__":{"format":"pt"}})",
        gateUp);
    std::vector<std::byte> downRouter = bf16Bytes({
        1, 0, 0, 1, 1, 0, 0, 1,
        3, 0, 0, 1});
    writeSafeTensor(root / "model-00002-of-00002.safetensors",
        R"({"model.layers.0.mlp.experts.down_proj":{"dtype":"BF16","shape":[2,2,2],"data_offsets":[0,16]},"model.layers.0.mlp.gate.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[16,24]},"__metadata__":{"format":"pt"}})",
        downRouter);
    std::ofstream(root / "model.safetensors.index.json")
        << R"({"metadata":{"total_size":56},"weight_map":{"model.layers.0.mlp.experts.gate_up_proj":"model-00001-of-00002.safetensors","model.layers.0.mlp.experts.down_proj":"model-00002-of-00002.safetensors","model.layers.0.mlp.gate.weight":"model-00002-of-00002.safetensors"}})";
}

hypermoe::models::ExpertWeightMap makeMappings(
    const hypermoe::models::ModelManifest& manifest) {
    using namespace hypermoe;
    models::ExpertWeightMap result;
    for (const auto& expert : manifest.experts) {
        const auto add = [&](models::ExpertWeightType type,
                             const models::ProjectionLocation& projection) {
            const auto* tensor = manifest.findTensor(projection.tensorName);
            result.add(expert.layerId, expert.expertId, type,
                {tensor->name, projection.shape, tensor->dtype, std::nullopt,
                 projection.offset, projection.size, expert.layerId, expert.expertId});
        };
        add(models::ExpertWeightType::GATE, expert.gate);
        add(models::ExpertWeightType::UP, expert.up);
        add(models::ExpertWeightType::DOWN, expert.down);
    }
    return result;
}

void testShardValidationAndPrecision() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    writeShardedQwen(temporary.path());
    importer::SafeTensorShardManager shards(temporary.path());
    expect(shards.shards().size() == 2 && shards.tensors().size() == 3 &&
               shards.find("model.layers.0.mlp.gate.weight")->sourceFile ==
                   "model-00002-of-00002.safetensors" &&
               shards.readRange("model.layers.0.mlp.gate.weight", 0, 2).size() == 2,
           "shard manager builds a global index and range-reads the resolved shard");
    importer::qwen::QwenImporter importer;
    const auto manifest = importer.inspect(temporary.path());
    const auto validation = importer::validation::CheckpointValidator::validate(
        temporary.path(), manifest);
    expect(validation.shardCount == 2 && validation.tensorCount == 3 &&
               validation.expertCount == 2 && validation.dtypeCounts.at("BF16") == 3,
           "checkpoint validator matches manifest ranges, shapes, dtypes, and mappings");
    auto invalid = manifest;
    invalid.tensors.front().size -= 2;
    expectThrows([&] {
        (void)importer::validation::CheckpointValidator::validate(
            temporary.path(), invalid);
    }, "checkpoint validator rejects manifest/source metadata drift");

    const std::vector<std::byte> fp16{
        std::byte{0x00}, std::byte{0x3c}, std::byte{0x00}, std::byte{0xc0},
        std::byte{0x01}, std::byte{0x00}};
    const auto fp16Values = tensor::precision::DTypeConverter::toFp32(
        fp16, tensor::DType::FP16);
    const auto bf16Values = tensor::precision::DTypeConverter::toFp32(
        bf16Bytes({1.0F, -2.5F}), tensor::DType::BF16);
    expect(fp16Values[0] == 1.0F && fp16Values[1] == -2.0F &&
               fp16Values[2] > 0.0F && fp16Values[2] < 1.0e-6F &&
               bf16Values == std::vector<float>({1.0F, -2.5F}),
           "dtype converter handles FP16 normals/subnormals and BF16 values");
    expectThrows([&] {
        (void)tensor::precision::DTypeConverter::toFp32(
            std::span<const std::byte>(fp16).first(1), tensor::DType::FP16);
    }, "dtype converter rejects misaligned source ranges");

    tensor::quantization::QuantizationPolicy q8{
        tensor::quantization::QuantizedDType::Q8, 0.125F, 0, 2};
    q8.validate({2, 2});
    tensor::quantization::QuantizationPolicy int8{
        tensor::quantization::QuantizedDType::INT8, 0.25F, -3, 4};
    tensor::quantization::QuantizationPolicy q4{
        tensor::quantization::QuantizedDType::Q4, 0.5F, 1, 2};
    int8.validate({2, 2});
    q4.validate({2, 2});
    expect(q8.toJson().find("\"Q8\"") != std::string::npos &&
               int8.toJson().find("\"zero_point\":-3") != std::string::npos &&
               q4.toJson().find("\"group_size\":2") != std::string::npos &&
               tensor::quantization::storageSizeBytes(
                   {2, 2}, tensor::quantization::QuantizedDType::Q8) == 4,
           "Q8 policy records scale, zero point, group size, and storage type");
    q8.groupSize = 3;
    expectThrows([&] { q8.validate({2, 2}); },
                 "quantization policy rejects incompatible group metadata");

    std::ofstream(temporary.path() / "model.safetensors.index.json",
                  std::ios::trunc)
        << R"({"weight_map":{"model.layers.0.mlp.experts.gate_up_proj":"model-00002-of-00002.safetensors","model.layers.0.mlp.experts.down_proj":"model-00002-of-00002.safetensors","model.layers.0.mlp.gate.weight":"model-00002-of-00002.safetensors"}})";
    expectThrows([&] {
        importer::SafeTensorShardManager invalid(temporary.path());
        (void)invalid;
    },
                 "shard manager rejects inconsistent Hugging Face weight maps");
}

void testRealBf16MoELayer() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(source);
    writeShardedQwen(source);
    importer::qwen::QwenImporter importer;
    const auto packing = conversion::ExpertPacker{}.pack(
        importer.inspect(source), source, packed);
    expect(packing.validationPassed && packing.shardCount == 2 &&
               packing.parametersIndexed == 28 &&
               std::filesystem::exists(packed / "conversion_report.json"),
           "model conversion reports shards, parameters, dtype statistics, and validation");

    const auto manifest = models::ModelManifest::load(packed / "manifest.json");
    auto store = std::make_shared<storage::ExpertStore>(packed);
    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto compute = std::make_shared<backend::CpuBackend>();
    auto transfers = std::make_shared<TransferManager>(loader, compute, 2);
    MemoryManager memory(4096, 4096);
    ExpertManager experts(memory, std::make_unique<LruCachePolicy>(), transfers);
    auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 2);
    for (const auto& record : store->index().records()) {
        experts.registerExpert({record.expert_id, record.layer_id,
            static_cast<std::size_t>(record.size), QuantizationType::Bf16,
            MemoryTier::Nvme});
        scheduler->registerExpert(record.layer_id, record.expert_id);
    }
    auto tensors = std::make_shared<tensor::CpuTensorBackend>();
    auto routerBackend = std::make_shared<router::CpuRouterBackend>();
    auto runtimeRouter = std::make_shared<router::Router>(
        manifest.router.config, routerBackend);
    auto executor = std::make_shared<ExpertMlpExecutor>(tensors);
    auto runtime = std::make_shared<runtime::MoERuntime>(
        runtimeRouter, scheduler, experts, makeMappings(manifest), tensors, executor);
    transformer::MoELayer layer(runtime, tensors);

    auto hidden = tensors->allocateTensor({1, 2}, tensor::DType::FP32);
    const std::vector<float> hiddenValues{1.0F, 2.0F};
    std::memcpy(hidden.data(), hiddenValues.data(), hiddenValues.size() * sizeof(float));
    const auto* routerMetadata = manifest.findTensor(
        manifest.router.tensors.front().tensorName);
    std::ifstream data(packed / "experts.bin", std::ios::binary);
    data.seekg(static_cast<std::streamoff>(routerMetadata->offset));
    std::vector<std::byte> routerBytes(static_cast<std::size_t>(routerMetadata->size));
    data.read(reinterpret_cast<char*>(routerBytes.data()),
              static_cast<std::streamsize>(routerBytes.size()));
    const auto routerValues = tensor::precision::DTypeConverter::toFp32(
        routerBytes, routerMetadata->dtype);
    auto routerWeights = tensors->allocateTensor(routerMetadata->shape,
                                                  tensor::DType::FP32);
    std::memcpy(routerWeights.data(), routerValues.data(),
                routerValues.size() * sizeof(float));

    const auto result = layer.execute(0, hidden, routerWeights);
    const std::vector<float> identity{1, 0, 0, 1};
    const auto trace = validation::CorrectnessOracle::expertMlpTrace(
        hiddenValues, 1, 2, identity, identity, identity, 2);
    std::vector<float> expected{hiddenValues[0] + trace.output[0],
                                hiddenValues[1] + trace.output[1]};
    const auto actual = std::span<const float>(
        static_cast<const float*>(result.output.data()), 2);
    const std::vector<float> routerIo{3, 0, 0, 1};
    const auto logits = validation::CorrectnessOracle::routerLogits(
        hiddenValues, routerIo, 2);
    const auto oracleRoute = validation::CorrectnessOracle::route(
        0, hiddenValues, routerIo, manifest.router.config);
    expect(logits == std::vector<float>({3, 2}) &&
               result.routing.selectedExpertIds == oracleRoute.selectedExpertIds &&
               validation::CorrectnessOracle::compare(
                   actual, expected,
                   validation::CorrectnessOracle::toleranceFor(
                       tensor::DType::BF16)).matches,
           "BF16 artifact executes router, selected expert, identity attention, and residual correctly");
    expect(trace.gateProjection == hiddenValues && trace.upProjection == hiddenValues &&
               trace.activation.size() == 2 && trace.gated.size() == 2,
           "correctness oracle exposes projection, activation, gated, and final stages");
}

} // namespace

int main() {
    testShardValidationAndPrecision();
    testRealBf16MoELayer();
    if (failures != 0) {
        std::cerr << failures << " Phase 10 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 10 real MoE layer tests passed\n";
    return EXIT_SUCCESS;
}
