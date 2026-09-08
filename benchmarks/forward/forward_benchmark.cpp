#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "models/runtime/ModelRuntime.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
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
            ("hypermoe-forward-benchmark-" +
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
    std::vector<float> result(size * size, 0.0F);
    for (std::size_t index = 0; index < size; ++index) {
        result[index * size + index] = 1.0F;
    }
    return result;
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
        throw std::invalid_argument("benchmark tensor shape mismatch");
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
        constexpr std::size_t layers = 2;
        constexpr std::size_t hidden = 32;
        constexpr std::size_t expertsPerLayer = 4;
        constexpr std::size_t vocabulary = 512;
        constexpr std::size_t tokenCount = 16;
        constexpr std::uint32_t iterations = 50;
        const auto reportPath = std::filesystem::path(
            argc > 1 ? argv[1] : "forward_report.json");
        TemporaryDirectory temporary;
        const auto matrix = identity(hidden);
        std::vector<storage::ExpertBlob> blobs;
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            for (std::uint32_t expert = 0; expert < expertsPerLayer; ++expert) {
                std::vector<float> packed;
                for (std::size_t projection = 0; projection < 3; ++projection) {
                    packed.insert(packed.end(), matrix.begin(), matrix.end());
                }
                blobs.push_back({layer, expert, 0, bytes(packed)});
            }
        }
        storage::ExpertStore::create(temporary.path(), blobs,
                                     R"({"benchmark":"phase14_forward"})");
        auto store = std::make_shared<storage::ExpertStore>(temporary.path());

        models::ModelManifest manifest;
        manifest.modelName = "Phase 14 forward benchmark fixture";
        manifest.architecture = models::ModelArchitecture::QWEN_MOE;
        manifest.sourceArchitecture = "Qwen3MoeForCausalLM";
        manifest.config.modelName = manifest.modelName;
        manifest.config.layerCount = layers;
        manifest.config.expertCount = expertsPerLayer;
        manifest.config.hiddenSize = hidden;
        manifest.config.intermediateSize = hidden;
        manifest.config.capabilities = {true, true, true, true, false, false};
        manifest.router.config = {
            expertsPerLayer, 1, router::RoutingNormalization::Softmax, true};
        manifest.router.layout = models::TensorLayout::InputOutput;
        models::runtime::ModelArchitecture architecture;
        architecture.layerCount = layers;
        architecture.hiddenDimension = hidden;
        architecture.attentionHeads = 4;
        architecture.keyValueHeads = 2;
        architecture.headDimension = 8;
        architecture.expertCount = expertsPerLayer;
        architecture.topK = 1;
        architecture.vocabularySize = vocabulary;
        manifest.runtimeArchitecture = architecture;

        models::ExpertWeightMap expertMappings;
        for (const auto& record : store->index().records()) {
            models::ManifestExpertMapping mapping;
            mapping.layerId = record.layer_id;
            mapping.expertId = record.expert_id;
            const auto add = [&](std::string role, std::uint64_t relativeOffset,
                                 models::ExpertWeightType type,
                                 models::ProjectionLocation& location) {
                const auto name = "layers." + std::to_string(record.layer_id) +
                    ".experts." + std::to_string(record.expert_id) + "." + role;
                const auto size = static_cast<std::uint64_t>(
                    hidden * hidden * sizeof(float));
                const auto offset = record.offset + relativeOffset;
                manifest.tensors.push_back(
                    {name, "experts.bin", offset, size, tensor::DType::FP32,
                     {hidden, hidden}});
                location = {name, offset, size, {hidden, hidden},
                            models::TensorLayout::InputOutput};
                expertMappings.add(record.layer_id, record.expert_id, type,
                    {name, {hidden,hidden}, tensor::DType::FP32, std::nullopt,
                     offset, size, record.layer_id, record.expert_id});
            };
            const auto width = static_cast<std::uint64_t>(
                hidden * hidden * sizeof(float));
            add("gate", 0, models::ExpertWeightType::GATE, mapping.gate);
            add("up", width, models::ExpertWeightType::UP, mapping.up);
            add("down", 2 * width, models::ExpertWeightType::DOWN, mapping.down);
            manifest.experts.push_back(std::move(mapping));
        }

        auto backend = std::make_shared<tensor::CpuTensorBackend>();
        models::runtime::RuntimeTensorMap tensors;
        const std::vector<float> norm(hidden, 1.0F);
        std::vector<float> routerWeights(hidden * expertsPerLayer);
        for (std::size_t feature = 0; feature < hidden; ++feature) {
            routerWeights[feature * expertsPerLayer + feature % expertsPerLayer] = 1.0F;
        }
        const auto addTensor = [&](std::string name, const tensor::Shape& shape,
                                   std::span<const float> data) {
            manifest.tensors.push_back(
                {name, "experts.bin", 0,
                 static_cast<std::uint64_t>(data.size_bytes()),
                 tensor::DType::FP32, shape});
            tensors.add(name, makeTensor(backend, shape, data));
            return name;
        };
        for (std::size_t layer = 0; layer < layers; ++layer) {
            models::ManifestLayerMapping mapping;
            mapping.layerId = static_cast<std::uint32_t>(layer);
            const auto prefix = "layers." + std::to_string(layer) + ".";
            mapping.queryProjection = {
                addTensor(prefix + "q_proj", {hidden,hidden}, matrix),
                models::TensorLayout::InputOutput};
            mapping.keyProjection = {
                addTensor(prefix + "k_proj", {hidden,16},
                          std::span<const float>(matrix).first(hidden * 16)),
                models::TensorLayout::InputOutput};
            mapping.valueProjection = {
                addTensor(prefix + "v_proj", {hidden,16},
                          std::span<const float>(matrix).first(hidden * 16)),
                models::TensorLayout::InputOutput};
            mapping.outputProjection = {
                addTensor(prefix + "o_proj", {hidden,hidden}, matrix),
                models::TensorLayout::InputOutput};
            mapping.inputNormTensor = addTensor(prefix + "input_norm", {hidden}, norm);
            mapping.postAttentionNormTensor =
                addTensor(prefix + "post_attention_norm", {hidden}, norm);
            mapping.routerTensor = addTensor(
                prefix + "router", {hidden,expertsPerLayer}, routerWeights);
            manifest.router.tensors.push_back(
                {static_cast<std::uint32_t>(layer), mapping.routerTensor});
            manifest.layers.push_back(std::move(mapping));
        }
        std::vector<float> embedding(vocabulary * hidden);
        std::vector<float> head(hidden * vocabulary);
        for (std::size_t token = 0; token < vocabulary; ++token) {
            for (std::size_t feature = 0; feature < hidden; ++feature) {
                const auto value = static_cast<float>((token + feature) % 17) / 17.0F;
                embedding[token * hidden + feature] = value;
                head[feature * vocabulary + token] = value;
            }
        }
        const auto embeddingName = addTensor(
            "model.token_embedding", {vocabulary,hidden}, embedding);
        const auto finalNormName = addTensor("model.final_norm", {hidden}, norm);
        const auto headName = addTensor("model.lm_head", {hidden,vocabulary}, head);
        manifest.modelIO = models::ManifestModelIO{
            embeddingName, finalNormName,
            {headName, models::TensorLayout::InputOutput}, false};
        manifest.validate();

        auto loader = std::make_shared<storage::DiskLoader>(store);
        auto transfers = std::make_shared<TransferManager>(
            loader, std::make_shared<backend::CpuBackend>(), 2);
        MemoryManager memory(64U << 20U, 64U << 20U);
        ExpertManager expertManager(
            memory, std::make_unique<LruCachePolicy>(), transfers);
        auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 2);
        for (const auto& record : store->index().records()) {
            expertManager.registerExpert(
                {record.expert_id, record.layer_id,
                 static_cast<std::size_t>(record.size), QuantizationType::Fp32,
                 MemoryTier::Nvme});
            scheduler->registerExpert(record.layer_id, record.expert_id);
        }
        auto router = std::make_shared<router::Router>(
            manifest.router.config, std::make_shared<router::CpuRouterBackend>());
        auto moeRuntime = std::make_shared<runtime::MoERuntime>(
            router, scheduler, expertManager, std::move(expertMappings), backend,
            std::make_shared<ExpertMlpExecutor>(backend));
        auto normalization = std::make_shared<transformer::norm::RMSNorm>(
            backend, hidden, architecture.inputNormalization.epsilon);
        auto transformerRuntime =
            std::make_shared<models::runtime::TransformerModelRuntime>(
                manifest, std::move(tensors),
                std::make_shared<transformer::attention::CpuAttention>(backend),
                normalization, normalization,
                std::make_shared<transformer::MoELayer>(moeRuntime, backend), backend);
        models::runtime::ModelRuntime model(manifest, transformerRuntime, backend);
        std::vector<std::uint32_t> ids(tokenCount);
        for (std::size_t index = 0; index < tokenCount; ++index) {
            ids[index] = static_cast<std::uint32_t>(index % vocabulary);
        }
        runtime::InferenceContext context;
        context.batchSize = tokenCount;
        context.hiddenDimension = hidden;
        (void)model.forward(context, ids);

        models::runtime::ModelForwardTimings totals;
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            context.sequencePosition = 0;
            const auto result = model.forward(context, ids);
            totals.embedding += result.timings.embedding;
            totals.transformer += result.timings.transformer;
            totals.finalNormalization += result.timings.finalNormalization;
            totals.lmHead += result.timings.lmHead;
            totals.total += result.timings.total;
        }
        const auto divisor = static_cast<double>(iterations);
        const auto cudaAvailable = tensor::CudaTensorBackend{}.available();
        std::ofstream report(reportPath);
        report << std::fixed << std::setprecision(6)
               << "{\n  \"benchmark\": \"phase14_forward\",\n"
               << "  \"fixture\": \"cpu_reference_2_layer_moe\",\n"
               << "  \"iterations\": " << iterations << ",\n"
               << "  \"tokens\": " << tokenCount << ",\n"
               << "  \"vocabulary_size\": " << vocabulary << ",\n"
               << "  \"cpu\": {\n"
               << "    \"embedding_ms\": " << milliseconds(totals.embedding) / divisor << ",\n"
               << "    \"transformer_ms\": " << milliseconds(totals.transformer) / divisor << ",\n"
               << "    \"final_norm_ms\": " << milliseconds(totals.finalNormalization) / divisor << ",\n"
               << "    \"lm_head_ms\": " << milliseconds(totals.lmHead) / divisor << ",\n"
               << "    \"total_forward_ms\": " << milliseconds(totals.total) / divisor << "\n  },\n"
               << "  \"cuda\": {\"available\":"
               << (cudaAvailable ? "true" : "false")
               << ",\"executed\":false,\"reason\":\"Phase 14 forward components are CPU reference implementations\"}\n}\n";
        if (!report) throw std::runtime_error("failed writing forward benchmark report");
        std::cout << "CPU forward: " << milliseconds(totals.total) / divisor
                  << " ms; report: " << reportPath << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Forward benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
