#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "models/ModelManifest.hpp"
#include "models/runtime/TransformerModelRuntime.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "runtime/cache/KVCache.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/norm/RMSNorm.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-model-runtime-benchmark-" +
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

std::vector<float> identity(std::size_t size) {
    std::vector<float> values(size * size, 0.0F);
    for (std::size_t index = 0; index < size; ++index) {
        values[index * size + index] = 1.0F;
    }
    return values;
}

std::vector<std::byte> bytes(std::span<const float> values) {
    std::vector<std::byte> result(values.size_bytes());
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

hypermoe::tensor::Tensor makeTensor(
    const std::shared_ptr<hypermoe::tensor::CpuTensorBackend>& backend,
    const hypermoe::tensor::Shape& shape,
    std::span<const float> values) {
    auto result = backend->allocateTensor(shape, hypermoe::tensor::DType::FP32);
    if (result.bytes() != values.size_bytes()) {
        throw std::invalid_argument("benchmark values do not match tensor shape");
    }
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

double milliseconds(std::chrono::nanoseconds duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

} // namespace

int main(int argc, char** argv) {
    try {
        using namespace hypermoe;
        constexpr std::size_t layerCount = 2;
        constexpr std::size_t hidden = 4;
        constexpr std::size_t tokenCount = 8;
        constexpr std::size_t expertCount = 2;
        const auto reportPath = std::filesystem::path(
            argc > 1 ? argv[1] : "model_runtime_report.json");
        TemporaryDirectory temporary;
        const auto matrix = identity(hidden);
        std::vector<storage::ExpertBlob> blobs;
        for (std::uint32_t layer = 0; layer < layerCount; ++layer) {
            for (std::uint32_t expert = 0; expert < expertCount; ++expert) {
                std::vector<float> packed;
                for (std::size_t projection = 0; projection < 3; ++projection) {
                    packed.insert(packed.end(), matrix.begin(), matrix.end());
                }
                blobs.push_back({layer, expert, 0, bytes(packed)});
            }
        }
        storage::ExpertStore::create(
            temporary.path(), blobs,
            R"({"benchmark":"phase13_model_runtime"})");
        auto store = std::make_shared<storage::ExpertStore>(temporary.path());

        models::ModelManifest manifest;
        manifest.modelName = "Phase13 benchmark fixture";
        manifest.architecture = models::ModelArchitecture::QWEN_MOE;
        manifest.sourceArchitecture = "Qwen3MoeForCausalLM";
        manifest.config.modelName = manifest.modelName;
        manifest.config.layerCount = layerCount;
        manifest.config.expertCount = expertCount;
        manifest.config.hiddenSize = hidden;
        manifest.config.intermediateSize = hidden;
        manifest.config.capabilities = {true, true, true, true, false, false};
        manifest.router.config = {
            expertCount, 1, router::RoutingNormalization::Softmax, true};
        manifest.router.layout = models::TensorLayout::InputOutput;
        models::runtime::ModelArchitecture architecture;
        architecture.layerCount = layerCount;
        architecture.hiddenDimension = hidden;
        architecture.attentionHeads = 2;
        architecture.keyValueHeads = 1;
        architecture.headDimension = 2;
        architecture.expertCount = expertCount;
        architecture.topK = 1;
        manifest.runtimeArchitecture = architecture;

        models::ExpertWeightMap expertMappings;
        for (const auto& record : store->index().records()) {
            models::ManifestExpertMapping mapping;
            mapping.layerId = record.layer_id;
            mapping.expertId = record.expert_id;
            const auto addProjection = [&](std::string role,
                                           std::uint64_t relativeOffset,
                                           models::ExpertWeightType type,
                                           models::ProjectionLocation& location) {
                const auto name = "layers." + std::to_string(record.layer_id) +
                                  ".experts." + std::to_string(record.expert_id) +
                                  "." + role;
                const auto offset = record.offset + relativeOffset;
                const auto size = static_cast<std::uint64_t>(
                    hidden * hidden * sizeof(float));
                manifest.tensors.push_back(
                    {name, "experts.bin", offset, size, tensor::DType::FP32,
                     {hidden, hidden}});
                location = {name, offset, size, {hidden, hidden},
                            models::TensorLayout::InputOutput};
                expertMappings.add(
                    record.layer_id, record.expert_id, type,
                    {name, {hidden, hidden}, tensor::DType::FP32, std::nullopt,
                     offset, size, record.layer_id, record.expert_id});
            };
            const auto matrixBytes = static_cast<std::uint64_t>(
                hidden * hidden * sizeof(float));
            addProjection("gate", 0, models::ExpertWeightType::GATE, mapping.gate);
            addProjection("up", matrixBytes, models::ExpertWeightType::UP, mapping.up);
            addProjection("down", 2 * matrixBytes,
                          models::ExpertWeightType::DOWN, mapping.down);
            manifest.experts.push_back(std::move(mapping));
        }

        auto tensors = std::make_shared<tensor::CpuTensorBackend>();
        models::runtime::RuntimeTensorMap runtimeTensors;
        const std::vector<float> kvWeights{
            1, 0,
            0, 1,
            0, 0,
            0, 0};
        const std::vector<float> routerWeights{
            2, 0,
            0, 2,
            0, 0,
            0, 0};
        const std::vector<float> normWeights(hidden, 1.0F);
        for (std::size_t layer = 0; layer < layerCount; ++layer) {
            models::ManifestLayerMapping mapping;
            mapping.layerId = static_cast<std::uint32_t>(layer);
            const auto add = [&](std::string role,
                                 const tensor::Shape& shape,
                                 std::span<const float> data) {
                const auto name = "layers." + std::to_string(layer) + "." + role;
                manifest.tensors.push_back(
                    {name, "experts.bin", 0,
                     static_cast<std::uint64_t>(data.size_bytes()),
                     tensor::DType::FP32, shape});
                runtimeTensors.add(name, makeTensor(tensors, shape, data));
                return name;
            };
            mapping.queryProjection = {
                add("q_proj", {hidden, hidden}, matrix),
                models::TensorLayout::InputOutput};
            mapping.keyProjection = {
                add("k_proj", {hidden, 2}, kvWeights),
                models::TensorLayout::InputOutput};
            mapping.valueProjection = {
                add("v_proj", {hidden, 2}, kvWeights),
                models::TensorLayout::InputOutput};
            mapping.outputProjection = {
                add("o_proj", {hidden, hidden}, matrix),
                models::TensorLayout::InputOutput};
            mapping.inputNormTensor = add("input_norm", {hidden}, normWeights);
            mapping.postAttentionNormTensor =
                add("post_attention_norm", {hidden}, normWeights);
            mapping.routerTensor = add("router", {hidden, expertCount}, routerWeights);
            manifest.router.tensors.push_back(
                {static_cast<std::uint32_t>(layer), mapping.routerTensor});
            manifest.layers.push_back(std::move(mapping));
        }
        manifest.validate();

        auto loader = std::make_shared<storage::DiskLoader>(store);
        auto transfers = std::make_shared<TransferManager>(
            loader, std::make_shared<backend::CpuBackend>(), 2);
        MemoryManager memory(1U << 20U, 1U << 20U);
        ExpertManager experts(memory, std::make_unique<LruCachePolicy>(), transfers);
        auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 2);
        for (const auto& record : store->index().records()) {
            experts.registerExpert({record.expert_id, record.layer_id,
                                    static_cast<std::size_t>(record.size),
                                    QuantizationType::Fp32, MemoryTier::Nvme});
            scheduler->registerExpert(record.layer_id, record.expert_id);
        }
        auto router = std::make_shared<router::Router>(
            manifest.router.config, std::make_shared<router::CpuRouterBackend>());
        auto moeRuntime = std::make_shared<runtime::MoERuntime>(
            router, scheduler, experts, std::move(expertMappings), tensors,
            std::make_shared<ExpertMlpExecutor>(tensors));
        auto cache = std::make_shared<runtime::cache::KVCache>(2, tokenCount, 1, 2);
        auto normalization =
            std::make_shared<transformer::norm::RMSNorm>(tensors, hidden);
        models::runtime::TransformerModelRuntime model(
            manifest, std::move(runtimeTensors),
            std::make_shared<transformer::attention::CpuAttention>(tensors),
            normalization, normalization,
            std::make_shared<transformer::MoELayer>(moeRuntime, tensors), tensors,
            cache);

        const std::vector<float> inputValues{
            1,0,0,1, 0,1,1,0, 1,1,0,0, 0.5F,0.25F,1,0,
            -1,1,0,0, 2,0.5F,0,1, 0,1,-1,1, 1,0.5F,0.25F,0};
        auto input = makeTensor(tensors, {tokenCount, hidden}, inputValues);
        runtime::InferenceContext context;
        context.batchSize = tokenCount;
        context.hiddenDimension = hidden;
        context.layerIndex = 0;
        cache->reset();
        (void)model.execute(context, input);

        constexpr std::uint32_t iterations = 100;
        std::chrono::nanoseconds attentionTime{};
        std::chrono::nanoseconds normTime{};
        std::chrono::nanoseconds routingTime{};
        std::chrono::nanoseconds expertTime{};
        std::chrono::nanoseconds combinationTime{};
        std::chrono::nanoseconds layerTime{};
        std::chrono::nanoseconds modelTime{};
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            cache->reset();
            context.sequencePosition = 0;
            const auto result = model.execute(context, input);
            modelTime += result.totalTime;
            for (const auto& layer : result.layers) {
                attentionTime += layer.timings.attention;
                normTime += layer.timings.normalization;
                routingTime += layer.execution.routingTime;
                expertTime += layer.execution.expertExecutionTime;
                combinationTime += layer.execution.expertCombinationTime;
                layerTime += layer.timings.total;
            }
        }
        const auto layerDivisor = static_cast<double>(iterations * layerCount);
        const auto modelDivisor = static_cast<double>(iterations);
        const auto memorySnapshot = memory.snapshot();
        const auto cudaAvailable = tensor::CudaTensorBackend{}.available();
        std::ofstream report(reportPath);
        report << std::fixed << std::setprecision(6)
               << "{\n  \"benchmark\": \"phase13_model_runtime\",\n"
               << "  \"fixture\": \"reduced_cpu_two_layer_moe\",\n"
               << "  \"iterations\": " << iterations << ",\n"
               << "  \"cpu\": {\n"
               << "    \"average_layer_ms\": "
               << milliseconds(layerTime) / layerDivisor << ",\n"
               << "    \"attention_ms_per_layer\": "
               << milliseconds(attentionTime) / layerDivisor << ",\n"
               << "    \"normalization_ms_per_layer\": "
               << milliseconds(normTime) / layerDivisor << ",\n"
               << "    \"routing_ms_per_layer\": "
               << milliseconds(routingTime) / layerDivisor << ",\n"
               << "    \"expert_ms_per_layer\": "
               << milliseconds(expertTime) / layerDivisor << ",\n"
               << "    \"expert_combination_ms_per_layer\": "
               << milliseconds(combinationTime) / layerDivisor << ",\n"
               << "    \"model_ms\": " << milliseconds(modelTime) / modelDivisor
               << "\n  },\n"
               << "  \"memory\": {\n"
               << "    \"shared_tensor_bytes\": "
               << model.tensors().memoryUsageBytes() << ",\n"
               << "    \"kv_cache_bytes\": " << cache->memoryUsageBytes() << ",\n"
               << "    \"logical_vram_bytes\": " << memorySnapshot.vram.usedBytes
               << ",\n    \"logical_ram_bytes\": " << memorySnapshot.ram.usedBytes
               << "\n  },\n"
               << "  \"cuda\": {\"available\":"
               << (cudaAvailable ? "true" : "false")
               << ",\"executed\":false,\"reason\":\"CUDA model layers are not part of the CPU reference benchmark\"}\n}\n";
        if (!report) throw std::runtime_error("failed writing model runtime report");
        std::cout << "Two-layer CPU model: "
                  << milliseconds(modelTime) / modelDivisor << " ms; report: "
                  << reportPath << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Model runtime benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
