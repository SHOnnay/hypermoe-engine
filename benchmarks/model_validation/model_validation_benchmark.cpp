#include "backend/CpuBackend.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "importer/SafeTensorShardManager.hpp"
#include "importer/SafeTensors.hpp"
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
#include "tools/model_convert/ExpertPacker.hpp"
#include "transformer/MoELayer.hpp"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
            ("hypermoe-model-validation-" + std::to_string(sequence.fetch_add(1)));
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

std::vector<std::byte> bf16(std::initializer_list<float> values) {
    std::vector<std::byte> bytes;
    for (const auto value : values) {
        const auto bits = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(value) >> 16U);
        bytes.push_back(static_cast<std::byte>(bits & 0xffU));
        bytes.push_back(static_cast<std::byte>(bits >> 8U));
    }
    return bytes;
}

void writeShard(const std::filesystem::path& path,
                std::string header,
                const std::vector<std::byte>& payload) {
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(path, std::ios::binary);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((length >> shift) & 0xffU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
}

void fixture(const std::filesystem::path& root) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Validation-Benchmark","num_hidden_layers":1,"num_experts":2,"hidden_size":2,"moe_intermediate_size":2,"num_experts_per_tok":1,"norm_topk_prob":true})";
    writeShard(root / "model-00001-of-00002.safetensors",
        R"({"model.layers.0.mlp.experts.gate_up_proj":{"dtype":"BF16","shape":[2,4,2],"data_offsets":[0,32]}})",
        bf16({1,0,0,1,1,0,0,1, 0.5F,0,0,0.5F,1,0,0,1}));
    writeShard(root / "model-00002-of-00002.safetensors",
        R"({"model.layers.0.mlp.experts.down_proj":{"dtype":"BF16","shape":[2,2,2],"data_offsets":[0,16]},"model.layers.0.mlp.gate.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[16,24]}})",
        bf16({1,0,0,1,1,0,0,1, 3,0,0,1}));
    std::ofstream(root / "model.safetensors.index.json")
        << R"({"metadata":{"total_size":56},"weight_map":{"model.layers.0.mlp.experts.gate_up_proj":"model-00001-of-00002.safetensors","model.layers.0.mlp.experts.down_proj":"model-00002-of-00002.safetensors","model.layers.0.mlp.gate.weight":"model-00002-of-00002.safetensors"}})";
}

hypermoe::models::ExpertWeightMap mappings(
    const hypermoe::models::ModelManifest& manifest) {
    using namespace hypermoe;
    models::ExpertWeightMap result;
    for (const auto& expert : manifest.experts) {
        const auto add = [&](models::ExpertWeightType type,
                             const models::ProjectionLocation& projection) {
            const auto* tensor = manifest.findTensor(projection.tensorName);
            if (!tensor) throw std::runtime_error("packed projection missing");
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

} // namespace

int main(int argc, char** argv) {
    try {
        TemporaryDirectory temporary;
        std::filesystem::path artifact;
        std::filesystem::path reportPath = "model_validation_report.json";
        bool reducedFixture = argc < 2;
        if (reducedFixture) {
            artifact = temporary.path() / "artifact";
            std::filesystem::create_directories(artifact);
            fixture(artifact);
        } else {
            artifact = argv[1];
        }
        if (argc >= 3) reportPath = argv[2];
        const auto packed = temporary.path() / "packed";

        const auto scanStart = Clock::now();
        const auto scanned = hypermoe::importer::SafeTensors::inspectArtifact(artifact);
        const auto scanEnd = Clock::now();
        const auto indexStart = Clock::now();
        hypermoe::importer::SafeTensorShardManager shardManager(artifact);
        const auto indexEnd = Clock::now();
        const auto manifestStart = Clock::now();
        hypermoe::importer::qwen::QwenImporter importer;
        const auto sourceManifest = importer.inspect(artifact);
        const auto manifestEnd = Clock::now();
        const auto validation =
            hypermoe::importer::validation::CheckpointValidator::validate(
                artifact, sourceManifest);
        const auto packStart = Clock::now();
        const auto packing = hypermoe::conversion::ExpertPacker{}.pack(
            sourceManifest,
            std::filesystem::is_directory(artifact) ? artifact : artifact.parent_path(),
            packed);
        const auto packEnd = Clock::now();

        using namespace hypermoe;
        const auto manifest = models::ModelManifest::load(packed / "manifest.json");
        auto store = std::make_shared<storage::ExpertStore>(packed);
        auto loader = std::make_shared<storage::DiskLoader>(store);
        auto compute = std::make_shared<backend::CpuBackend>();
        auto transfers = std::make_shared<TransferManager>(loader, compute, 2);
        std::size_t largestExpert{};
        for (const auto& record : store->index().records()) {
            largestExpert = std::max(largestExpert, static_cast<std::size_t>(record.size));
        }
        const auto capacityFactor = manifest.router.config.topK + 1;
        if (capacityFactor == 0 || largestExpert >
                std::numeric_limits<std::size_t>::max() / capacityFactor) {
            throw std::overflow_error("benchmark expert capacity overflow");
        }
        MemoryManager memory(largestExpert * capacityFactor,
                             largestExpert * capacityFactor);
        ExpertManager experts(memory, std::make_unique<LruCachePolicy>(), transfers);
        auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 2);
        for (const auto& record : store->index().records()) {
            experts.registerExpert({record.expert_id, record.layer_id,
                static_cast<std::size_t>(record.size),
                static_cast<QuantizationType>(record.quantization_type),
                MemoryTier::Nvme});
            scheduler->registerExpert(record.layer_id, record.expert_id);
        }
        auto tensorBackend = std::make_shared<tensor::CpuTensorBackend>();
        auto routerBackend = std::make_shared<router::CpuRouterBackend>();
        auto runtimeRouter = std::make_shared<router::Router>(
            manifest.router.config, routerBackend);
        auto executor = std::make_shared<ExpertMlpExecutor>(tensorBackend);
        auto runtime = std::make_shared<runtime::MoERuntime>(
            runtimeRouter, scheduler, experts, mappings(manifest),
            tensorBackend, executor);
        transformer::MoELayer layer(runtime, tensorBackend);
        auto hidden = tensorBackend->allocateTensor(
            {1, manifest.config.hiddenSize}, tensor::DType::FP32);
        std::fill_n(static_cast<float*>(hidden.data()), manifest.config.hiddenSize, 0.5F);
        const auto benchmarkLayer = manifest.router.tensors.front().layerId;
        const auto* routerMetadata = manifest.findTensor(
            manifest.router.tensors.front().tensorName);
        std::ifstream packedData(packed / "experts.bin", std::ios::binary);
        packedData.seekg(static_cast<std::streamoff>(routerMetadata->offset));
        std::vector<std::byte> routerBytes(static_cast<std::size_t>(routerMetadata->size));
        packedData.read(reinterpret_cast<char*>(routerBytes.data()),
                        static_cast<std::streamsize>(routerBytes.size()));
        const auto routerFp32 = tensor::precision::DTypeConverter::toFp32(
            routerBytes, routerMetadata->dtype);
        auto routerWeights = tensorBackend->allocateTensor(
            routerMetadata->shape, tensor::DType::FP32);
        std::memcpy(routerWeights.data(), routerFp32.data(),
                    routerFp32.size() * sizeof(float));

        const auto expertStart = Clock::now();
        const auto expertResult = runtime->executeLayer(
            benchmarkLayer, hidden, routerWeights);
        const auto expertEnd = Clock::now();
        const auto layerStart = Clock::now();
        const auto layerResult = layer.execute(benchmarkLayer, hidden, routerWeights);
        const auto layerEnd = Clock::now();
        if (!expertResult.routing.valid() || !layerResult.routing.valid()) {
            throw std::runtime_error("benchmark execution returned invalid routing");
        }

        std::ofstream report(reportPath, std::ios::trunc);
        report << std::fixed << std::setprecision(6)
               << "{\n  \"benchmark_kind\": \"cpu_correctness\",\n"
               << "  \"reduced_fixture\": " << (reducedFixture ? "true" : "false") << ",\n"
               << "  \"checkpoint_scan_ms\": " << ms(scanStart, scanEnd) << ",\n"
               << "  \"shard_indexing_ms\": " << ms(indexStart, indexEnd) << ",\n"
               << "  \"manifest_creation_ms\": " << ms(manifestStart, manifestEnd) << ",\n"
               << "  \"packing_ms\": " << ms(packStart, packEnd) << ",\n"
               << "  \"single_expert_path_ms\": " << ms(expertStart, expertEnd) << ",\n"
               << "  \"transformer_layer_ms\": " << ms(layerStart, layerEnd) << ",\n"
               << "  \"shards\": " << validation.shardCount << ",\n"
               << "  \"tensors\": " << scanned.size() << ",\n"
               << "  \"experts\": " << packing.experts << ",\n"
               << "  \"parameters_indexed\": " << packing.parametersIndexed << ",\n"
               << "  \"packed_bytes\": " << packing.bytesWritten << "\n}\n";
        if (!report) throw std::runtime_error("cannot write model validation report");
        std::cout << reportPath << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model validation benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
