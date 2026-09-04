#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "models/runtime/ModelArchitecture.hpp"
#include "models/runtime/TransformerModelRuntime.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "runtime/cache/KVCache.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tools/model_convert/ExpertPacker.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/norm/RMSNorm.hpp"
#include "transformer/position/RoPE.hpp"
#include "validation/CorrectnessOracle.hpp"

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
#include <set>
#include <span>
#include <sstream>
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
                ("hypermoe-phase13-" +
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

struct TensorDefinition {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

std::vector<float> identity(std::size_t size, float scale = 1.0F) {
    std::vector<float> result(size * size, 0.0F);
    for (std::size_t index = 0; index < size; ++index) {
        result[index * size + index] = scale;
    }
    return result;
}

void writeQwenArchitectureFixture(const std::filesystem::path& root) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Phase13-Complete-Fixture","num_hidden_layers":2,"num_experts":2,"hidden_size":4,"moe_intermediate_size":4,"num_experts_per_tok":1,"norm_topk_prob":true,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":0.000001,"rope_theta":10000})";
    std::vector<TensorDefinition> tensors;
    const auto fullIdentity = identity(4);
    const std::vector<float> keyValueProjection{
        1, 0, 0, 0,
        0, 1, 0, 0};
    const std::vector<float> router{
        2, 0, 0, 0,
        0, 2, 0, 0};
    const std::vector<float> norm{1, 1, 1, 1};
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const auto prefix = "model.layers." + std::to_string(layer);
        tensors.push_back({prefix + ".self_attn.q_proj.weight", {4, 4},
                           fullIdentity});
        tensors.push_back({prefix + ".self_attn.k_proj.weight", {2, 4},
                           keyValueProjection});
        tensors.push_back({prefix + ".self_attn.v_proj.weight", {2, 4},
                           keyValueProjection});
        tensors.push_back({prefix + ".self_attn.o_proj.weight", {4, 4},
                           fullIdentity});
        tensors.push_back({prefix + ".input_layernorm.weight", {4}, norm});
        tensors.push_back({prefix + ".post_attention_layernorm.weight", {4}, norm});
        tensors.push_back({prefix + ".mlp.gate.weight", {2, 4}, router});
        for (std::size_t expert = 0; expert < 2; ++expert) {
            const auto expertPrefix = prefix + ".mlp.experts." +
                                      std::to_string(expert);
            tensors.push_back({expertPrefix + ".gate_proj.weight", {4, 4},
                               fullIdentity});
            tensors.push_back({expertPrefix + ".up_proj.weight", {4, 4},
                               fullIdentity});
            tensors.push_back({expertPrefix + ".down_proj.weight", {4, 4},
                               fullIdentity});
        }
    }
    std::ostringstream header;
    std::vector<float> payload;
    header << '{';
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const auto& tensor = tensors[index];
        std::size_t elementCount{1};
        for (const auto dimension : tensor.shape) elementCount *= dimension;
        if (elementCount != tensor.values.size()) {
            throw std::logic_error("fixture tensor shape mismatch");
        }
        const auto begin = payload.size() * sizeof(float);
        payload.insert(payload.end(), tensor.values.begin(), tensor.values.end());
        const auto end = payload.size() * sizeof(float);
        if (index != 0) header << ',';
        header << '"' << tensor.name << "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
            if (dimension != 0) header << ',';
            header << tensor.shape[dimension];
        }
        header << "],\"data_offsets\":[" << begin << ',' << end << "]}";
    }
    header << '}';
    auto headerText = header.str();
    while (headerText.size() % 8 != 0) headerText.push_back(' ');
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    const auto headerSize = static_cast<std::uint64_t>(headerText.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((headerSize >> shift) & 0xffU));
    }
    output.write(headerText.data(), static_cast<std::streamsize>(headerText.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed writing Phase 13 fixture");
}

hypermoe::models::ExpertWeightMap makeExpertMappings(
    const hypermoe::models::ModelManifest& manifest) {
    using namespace hypermoe;
    models::ExpertWeightMap result;
    for (const auto& expert : manifest.experts) {
        const auto add = [&](models::ExpertWeightType type,
                             const models::ProjectionLocation& projection) {
            const auto* source = manifest.findTensor(projection.tensorName);
            if (!source) throw std::runtime_error("fixture projection is missing");
            result.add(expert.layerId, expert.expertId, type,
                       {source->name, projection.shape, source->dtype,
                        std::nullopt, projection.offset, projection.size,
                        expert.layerId, expert.expertId});
        };
        add(models::ExpertWeightType::GATE, expert.gate);
        add(models::ExpertWeightType::UP, expert.up);
        add(models::ExpertWeightType::DOWN, expert.down);
    }
    return result;
}

hypermoe::tensor::Tensor makeTensor(
    const std::shared_ptr<hypermoe::tensor::CpuTensorBackend>& backend,
    const hypermoe::tensor::Shape& shape,
    std::span<const float> values) {
    auto result = backend->allocateTensor(shape, hypermoe::tensor::DType::FP32);
    if (result.bytes() != values.size_bytes()) {
        throw std::invalid_argument("test tensor values do not match shape");
    }
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

std::vector<float> values(const hypermoe::tensor::Tensor& tensor) {
    const auto* first = static_cast<const float*>(tensor.data());
    return {first, first + tensor.shape().elementCount()};
}

hypermoe::models::runtime::RuntimeTensorMap loadRuntimeTensors(
    const hypermoe::models::ModelManifest& manifest,
    const std::filesystem::path& dataPath,
    const std::shared_ptr<hypermoe::tensor::CpuTensorBackend>& backend) {
    std::set<std::string> names;
    for (const auto& layer : manifest.layers) {
        names.insert(layer.queryProjection.tensorName);
        names.insert(layer.keyProjection.tensorName);
        names.insert(layer.valueProjection.tensorName);
        names.insert(layer.outputProjection.tensorName);
        names.insert(layer.inputNormTensor);
        names.insert(layer.postAttentionNormTensor);
        names.insert(layer.routerTensor);
    }
    std::ifstream input(dataPath, std::ios::binary);
    hypermoe::models::runtime::RuntimeTensorMap result;
    for (const auto& name : names) {
        const auto* metadata = manifest.findTensor(name);
        if (!metadata || metadata->dtype != hypermoe::tensor::DType::FP32) {
            throw std::runtime_error("runtime fixture tensor metadata is invalid");
        }
        input.seekg(static_cast<std::streamoff>(metadata->offset));
        std::vector<std::byte> bytes(static_cast<std::size_t>(metadata->size));
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!input) throw std::runtime_error("runtime fixture tensor read failed");
        auto tensor = backend->allocateTensor(metadata->shape, metadata->dtype);
        std::memcpy(tensor.data(), bytes.data(), bytes.size());
        result.add(name, std::move(tensor));
    }
    return result;
}

std::vector<float> rmsNorm(std::span<const float> input,
                           std::size_t tokens,
                           std::size_t hidden,
                           float epsilon) {
    std::vector<float> result(input.size());
    for (std::size_t token = 0; token < tokens; ++token) {
        double sum{};
        for (std::size_t feature = 0; feature < hidden; ++feature) {
            const auto value = input[token * hidden + feature];
            sum += static_cast<double>(value) * value;
        }
        const auto inverse = 1.0 / std::sqrt(
            sum / static_cast<double>(hidden) + epsilon);
        for (std::size_t feature = 0; feature < hidden; ++feature) {
            result[token * hidden + feature] = static_cast<float>(
                input[token * hidden + feature] * inverse);
        }
    }
    return result;
}

std::vector<float> referenceLayer(std::span<const float> input) {
    constexpr std::size_t tokens = 2;
    constexpr std::size_t hidden = 4;
    auto normalizedInput = rmsNorm(input, tokens, hidden, 1.0e-6F);
    const auto query = hypermoe::validation::CorrectnessOracle::applyRoPE(
        normalizedInput, tokens, 2, 2, 0, 10000.0F);
    std::vector<float> keyValue(tokens * 2);
    for (std::size_t token = 0; token < tokens; ++token) {
        keyValue[token * 2] = normalizedInput[token * hidden];
        keyValue[token * 2 + 1] = normalizedInput[token * hidden + 1];
    }
    const auto key = hypermoe::validation::CorrectnessOracle::applyRoPE(
        keyValue, tokens, 1, 2, 0, 10000.0F);
    const std::vector<std::uint64_t> positions{0, 1};
    const auto attention = hypermoe::validation::CorrectnessOracle::causalAttention(
        query, key, keyValue, positions, tokens, 2, 1, 2, 0);
    std::vector<float> attentionResidual(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        attentionResidual[index] = input[index] + attention[index];
    }
    const auto expertInput = rmsNorm(attentionResidual, tokens, hidden, 1.0e-6F);
    const auto expertOutput = hypermoe::validation::CorrectnessOracle::expertMlp(
        expertInput, tokens, hidden, identity(hidden), identity(hidden),
        identity(hidden), hidden);
    std::vector<float> result(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        result[index] = attentionResidual[index] + expertOutput[index];
    }
    return result;
}

void testRoPEAndKVCache() {
    using namespace hypermoe;
    std::vector<float> ropeValues{1, 0, 0, 1};
    transformer::position::RoPE rope(10000.0F);
    rope.apply(ropeValues, 2, 1, 2, 0);
    const auto oracle = validation::CorrectnessOracle::applyRoPE(
        std::vector<float>{1, 0, 0, 1}, 2, 1, 2, 0, 10000.0F);
    expect(validation::CorrectnessOracle::compare(
               ropeValues, oracle,
               validation::CorrectnessOracle::toleranceFor(
                   tensor::DType::FP32)).matches &&
               std::abs(ropeValues[2] + std::sin(1.0F)) < 1.0e-5F &&
               std::abs(ropeValues[3] - std::cos(1.0F)) < 1.0e-5F,
           "RoPE matches independent position-zero and position-one rotations");
    expectThrows([] { transformer::position::RoPE invalid(0.0F); },
                 "RoPE rejects invalid theta");

    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    const std::vector<float> attentionInputValues{
        1, 0, 0, 1,
        0, 1, 1, 0};
    const std::vector<float> keyValueWeights{
        1, 0,
        0, 1,
        0, 0,
        0, 0};
    auto attentionInput = makeTensor(backend, {2, 4}, attentionInputValues);
    auto queryWeights = makeTensor(backend, {4, 4}, identity(4));
    auto keyWeights = makeTensor(backend, {4, 2}, keyValueWeights);
    auto valueWeights = makeTensor(backend, {4, 2}, keyValueWeights);
    auto outputWeights = makeTensor(backend, {4, 4}, identity(4));
    transformer::attention::CpuAttention attention(backend);
    transformer::attention::AttentionConfiguration attentionConfiguration;
    attentionConfiguration.headCount = 2;
    attentionConfiguration.keyValueHeadCount = 1;
    attentionConfiguration.headDimension = 2;
    attentionConfiguration.causal = true;
    attentionConfiguration.rotaryEmbedding = true;
    const auto attentionResult = attention.execute(
        attentionInput,
        {queryWeights, keyWeights, valueWeights, outputWeights},
        attentionConfiguration);
    const auto referenceQuery = validation::CorrectnessOracle::applyRoPE(
        attentionInputValues, 2, 2, 2, 0, 10000.0F);
    const std::vector<float> referenceKeyValue{1, 0, 0, 1};
    const auto referenceKey = validation::CorrectnessOracle::applyRoPE(
        referenceKeyValue, 2, 1, 2, 0, 10000.0F);
    const auto referenceContext = validation::CorrectnessOracle::causalAttention(
        referenceQuery, referenceKey, referenceKeyValue,
        std::vector<std::uint64_t>{0, 1}, 2, 2, 1, 2, 0);
    const auto probabilities = values(attentionResult.probabilities);
    expect(validation::CorrectnessOracle::compare(
               values(attentionResult.context), referenceContext,
               validation::CorrectnessOracle::toleranceFor(
                   tensor::DType::FP32)).matches &&
               attentionResult.scores.shape() == tensor::Shape{2, 2, 2} &&
               probabilities[1] == 0.0F && probabilities[5] == 0.0F,
           "causal grouped-query attention masks future keys and matches the oracle");

    runtime::cache::KVCache cache(2, 4, 1, 2);
    auto keys = makeTensor(backend, {2, 1, 2},
                           std::vector<float>{1, 2, 3, 4});
    auto cachedValues = makeTensor(backend, {2, 1, 2},
                                   std::vector<float>{5, 6, 7, 8});
    cache.append(0, 0, keys, cachedValues);
    const auto snapshot = cache.snapshot(0);
    expect(snapshot.positions == std::vector<std::uint64_t>({0, 1}) &&
               snapshot.keys == std::vector<float>({1, 2, 3, 4}) &&
               snapshot.values == std::vector<float>({5, 6, 7, 8}) &&
               cache.tokenCount(1) == 0 && cache.memoryUsageBytes() > 0,
           "KV cache isolates layers and preserves positions, keys, and values");
    expectThrows([&] { cache.append(0, 3, keys, cachedValues); },
                 "KV cache rejects non-contiguous sequence positions");
    cache.clear(0);
    expect(cache.tokenCount(0) == 0, "KV cache clears one layer independently");
}

void testCompleteQwenRuntime() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(source);
    writeQwenArchitectureFixture(source);
    importer::qwen::QwenImporter importer;
    const auto sourceManifest = importer.inspect(source);
    const auto architecture =
        models::runtime::ModelArchitecture::fromManifest(sourceManifest);
    expect(sourceManifest.layers.size() == 2 && architecture.layerCount == 2 &&
               architecture.hiddenDimension == 4 &&
               architecture.attentionHeads == 2 &&
               architecture.keyValueHeads == 1 && architecture.headDimension == 2 &&
               sourceManifest.findLayer(1)->queryProjection.layout ==
                   models::TensorLayout::OutputInput,
           "Qwen importer maps complete architecture and transformer tensor roles");
    auto invalid = *sourceManifest.runtimeArchitecture;
    invalid.attentionHeads = 3;
    expectThrows([&] { invalid.validate(); },
                 "model architecture rejects incompatible head dimensions");

    const auto packing = conversion::ExpertPacker{}.pack(
        sourceManifest, source, packed);
    expect(packing.validationPassed,
           "complete Qwen architecture fixture validates and packs successfully");
    const auto manifest = models::ModelManifest::load(packed / "manifest.json");
    expect(manifest.layers.size() == 2 &&
               manifest.findLayer(0)->queryProjection.layout ==
                   models::TensorLayout::InputOutput,
           "packed manifest retains generic layer mappings in execution layout");

    auto store = std::make_shared<storage::ExpertStore>(packed);
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
    auto tensors = std::make_shared<tensor::CpuTensorBackend>();
    auto router = std::make_shared<router::Router>(
        manifest.router.config, std::make_shared<router::CpuRouterBackend>());
    auto expertExecutor = std::make_shared<ExpertMlpExecutor>(tensors);
    auto moeRuntime = std::make_shared<runtime::MoERuntime>(
        router, scheduler, experts, makeExpertMappings(manifest), tensors,
        expertExecutor);
    auto moe = std::make_shared<transformer::MoELayer>(moeRuntime, tensors);
    auto attention = std::make_shared<transformer::attention::CpuAttention>(tensors);
    auto normalization = std::make_shared<transformer::norm::RMSNorm>(
        tensors, architecture.hiddenDimension,
        architecture.postAttentionNormalization.epsilon);
    auto cache = std::make_shared<runtime::cache::KVCache>(
        architecture.layerCount, 8, architecture.keyValueHeads,
        architecture.headDimension);
    models::runtime::TransformerModelRuntime model(
        manifest, loadRuntimeTensors(manifest, store->dataPath(), tensors),
        attention, normalization, normalization, moe, tensors, cache);

    const std::vector<float> inputValues{
        1, 2, 0.5F, -0.5F,
        0.25F, 1.5F, -1, 0.75F};
    auto input = makeTensor(tensors, {2, 4}, inputValues);
    runtime::InferenceContext context;
    context.batchSize = 2;
    context.sequencePosition = 0;
    context.hiddenDimension = 4;
    context.layerIndex = 0;
    const auto result = model.execute(context, input);
    const auto layer0 = referenceLayer(inputValues);
    const auto layer1 = referenceLayer(layer0);
    std::vector<std::vector<float>> actualLayers;
    for (const auto& layer : result.layers) actualLayers.push_back(values(layer.output));
    const std::vector<std::vector<float>> expectedLayers{layer0, layer1};
    const auto comparison = validation::CorrectnessOracle::compareModelLayers(
        actualLayers, expectedLayers, tensor::DType::FP32);
    expect(comparison.matches() && values(result.output) == actualLayers.back() &&
               result.layers.size() == 2 &&
               result.layers[0].routing.size() == 2 &&
               result.layers[1].execution.expertAssignments == 2 &&
               cache->tokenCount(0) == 2 && cache->tokenCount(1) == 2,
           "multi-layer runtime matches independent attention, RoPE, MoE, residual, and cache references");
    expect(model.tensors().size() == 14 && model.tensors().memoryUsageBytes() > 0,
           "runtime tensor mapping owns all per-layer shared tensors without Qwen names in execution code");
}

} // namespace

int main() {
    testRoPEAndKVCache();
    testCompleteQwenRuntime();
    if (failures != 0) {
        std::cerr << failures << " Phase 13 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 13 model runtime tests passed\n";
    return EXIT_SUCCESS;
}
