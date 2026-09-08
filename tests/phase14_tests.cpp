#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "models/runtime/ModelRuntime.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tools/model_convert/ExpertPacker.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/embedding/Embedding.hpp"
#include "transformer/norm/RMSNorm.hpp"
#include "transformer/output/FinalNorm.hpp"
#include "transformer/output/LMHead.hpp"
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
                ("hypermoe-phase14-" +
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

std::vector<float> identity(std::size_t size) {
    std::vector<float> result(size * size, 0.0F);
    for (std::size_t index = 0; index < size; ++index) {
        result[index * size + index] = 1.0F;
    }
    return result;
}

const std::vector<float> embeddingWeights{
    1,0,0,0, 0,1,0,0, 0,0,1,0,
    0,0,0,1, 1,1,0,0, 0,0,1,1};
const std::vector<float> headWeights{
    1,0,0,0, 0,1,0,0, 0,0,1,0,
    0,0,0,1, 1,1,0,0, 0,0,1,1};
const std::vector<float> finalNormWeights{1, 0.5F, 1.5F, 2};

void writeFixture(const std::filesystem::path& root, bool tied = false) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Phase14-Forward-Fixture","num_hidden_layers":1,"num_experts":2,"hidden_size":4,"moe_intermediate_size":4,"num_experts_per_tok":1,"norm_topk_prob":true,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":0.000001,"rope_theta":10000,"vocab_size":6,"tie_word_embeddings":)"
        << (tied ? "true" : "false") << '}';
    const auto matrix = identity(4);
    const std::vector<float> keyValue{1,0,0,0, 0,1,0,0};
    const std::vector<float> router{2,0,0,0, 0,2,0,0};
    const std::vector<float> norm(4, 1.0F);
    std::vector<TensorDefinition> tensors{
        {"model.embed_tokens.weight", {6,4}, embeddingWeights},
        {"model.norm.weight", {4}, finalNormWeights},
        {"model.layers.0.self_attn.q_proj.weight", {4,4}, matrix},
        {"model.layers.0.self_attn.k_proj.weight", {2,4}, keyValue},
        {"model.layers.0.self_attn.v_proj.weight", {2,4}, keyValue},
        {"model.layers.0.self_attn.o_proj.weight", {4,4}, matrix},
        {"model.layers.0.input_layernorm.weight", {4}, norm},
        {"model.layers.0.post_attention_layernorm.weight", {4}, norm},
        {"model.layers.0.mlp.gate.weight", {2,4}, router}};
    if (!tied) {
        tensors.push_back({"lm_head.weight", {6,4}, headWeights});
    }
    for (std::size_t expert = 0; expert < 2; ++expert) {
        const auto prefix = "model.layers.0.mlp.experts." + std::to_string(expert);
        tensors.push_back({prefix + ".gate_proj.weight", {4,4}, matrix});
        tensors.push_back({prefix + ".up_proj.weight", {4,4}, matrix});
        tensors.push_back({prefix + ".down_proj.weight", {4,4}, matrix});
    }
    std::ostringstream header;
    std::vector<float> payload;
    header << '{';
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const auto& definition = tensors[index];
        if (index != 0) header << ',';
        const auto begin = payload.size() * sizeof(float);
        payload.insert(payload.end(), definition.values.begin(), definition.values.end());
        const auto end = payload.size() * sizeof(float);
        header << '"' << definition.name
               << "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < definition.shape.size(); ++dimension) {
            if (dimension != 0) header << ',';
            header << definition.shape[dimension];
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
    if (!output) throw std::runtime_error("failed writing Phase 14 fixture");
}

hypermoe::tensor::Tensor makeTensor(
    const std::shared_ptr<hypermoe::tensor::CpuTensorBackend>& backend,
    const hypermoe::tensor::Shape& shape,
    std::span<const float> source) {
    auto result = backend->allocateTensor(shape, hypermoe::tensor::DType::FP32);
    if (result.bytes() != source.size_bytes()) {
        throw std::invalid_argument("test tensor shape mismatch");
    }
    std::memcpy(result.data(), source.data(), source.size_bytes());
    return result;
}

std::vector<float> values(const hypermoe::tensor::Tensor& value) {
    const auto* first = static_cast<const float*>(value.data());
    return {first, first + value.shape().elementCount()};
}

hypermoe::models::ExpertWeightMap expertMappings(
    const hypermoe::models::ModelManifest& manifest) {
    using namespace hypermoe;
    models::ExpertWeightMap result;
    for (const auto& expert : manifest.experts) {
        const auto add = [&](models::ExpertWeightType type,
                             const models::ProjectionLocation& projection) {
            const auto* source = manifest.findTensor(projection.tensorName);
            if (!source) throw std::runtime_error("expert tensor mapping is absent");
            result.add(expert.layerId, expert.expertId, type,
                       {source->name, projection.shape, source->dtype, std::nullopt,
                        projection.offset, projection.size,
                        expert.layerId, expert.expertId});
        };
        add(models::ExpertWeightType::GATE, expert.gate);
        add(models::ExpertWeightType::UP, expert.up);
        add(models::ExpertWeightType::DOWN, expert.down);
    }
    return result;
}

hypermoe::models::runtime::RuntimeTensorMap loadRuntimeTensors(
    const hypermoe::models::ModelManifest& manifest,
    const std::filesystem::path& path,
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
    if (!manifest.modelIO) throw std::runtime_error("fixture has no model I/O mapping");
    names.insert(manifest.modelIO->tokenEmbeddingTensor);
    names.insert(manifest.modelIO->finalNormTensor);
    names.insert(manifest.modelIO->lmHead.tensorName);
    std::ifstream input(path, std::ios::binary);
    hypermoe::models::runtime::RuntimeTensorMap result;
    for (const auto& name : names) {
        const auto* metadata = manifest.findTensor(name);
        if (!metadata) throw std::runtime_error("runtime tensor metadata is absent");
        input.seekg(static_cast<std::streamoff>(metadata->offset));
        std::vector<std::byte> bytes(static_cast<std::size_t>(metadata->size));
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!input) throw std::runtime_error("runtime tensor load failed");
        auto tensor = backend->allocateTensor(metadata->shape, metadata->dtype);
        std::memcpy(tensor.data(), bytes.data(), bytes.size());
        result.add(name, std::move(tensor));
    }
    return result;
}

std::vector<float> referenceLayer(std::span<const float> input) {
    using hypermoe::validation::CorrectnessOracle;
    const std::vector<float> ones(4, 1.0F);
    auto normalized = CorrectnessOracle::rmsNorm(input, 2, 4, ones, 1.0e-6F);
    const auto query = CorrectnessOracle::applyRoPE(normalized, 2, 2, 2, 0, 10000);
    std::vector<float> keyValue(4);
    for (std::size_t token = 0; token < 2; ++token) {
        keyValue[token * 2] = normalized[token * 4];
        keyValue[token * 2 + 1] = normalized[token * 4 + 1];
    }
    const auto key = CorrectnessOracle::applyRoPE(keyValue, 2, 1, 2, 0, 10000);
    const auto attention = CorrectnessOracle::causalAttention(
        query, key, keyValue, std::vector<std::uint64_t>{0,1}, 2, 2, 1, 2, 0);
    std::vector<float> residual(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        residual[index] = input[index] + attention[index];
    }
    const auto expertInput = CorrectnessOracle::rmsNorm(
        residual, 2, 4, ones, 1.0e-6F);
    const auto expert = CorrectnessOracle::expertMlp(
        expertInput, 2, 4, identity(4), identity(4), identity(4), 4);
    for (std::size_t index = 0; index < residual.size(); ++index) {
        residual[index] += expert[index];
    }
    return residual;
}

void testForwardComponents() {
    using namespace hypermoe;
    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    auto embeddingTable = makeTensor(backend, {6,4}, embeddingWeights);
    transformer::embedding::Embedding embedding(backend, 6, 4);
    const std::vector<std::uint32_t> ids{1,5};
    const auto embedded = embedding.execute(ids, embeddingTable.view());
    const auto expectedEmbedding = validation::CorrectnessOracle::embedding(
        ids, embeddingWeights, 6, 4);
    expect(validation::CorrectnessOracle::compare(
               values(embedded), expectedEmbedding, {1.0e-5F, 1.0e-5F}).matches,
           "embedding lookup matches the independent oracle");
    expectThrows([&] {
        const std::vector<std::uint32_t> bad{6};
        (void)embedding.execute(bad, embeddingTable.view());
    }, "embedding rejects out-of-vocabulary token IDs");

    auto normWeight = makeTensor(backend, {4}, finalNormWeights);
    transformer::output::FinalNorm finalNorm(backend, 4, 1.0e-6F);
    const auto normalized = finalNorm.execute(embedded.view(), normWeight.view());
    const auto expectedNorm = validation::CorrectnessOracle::rmsNorm(
        expectedEmbedding, 2, 4, finalNormWeights, 1.0e-6F);
    expect(validation::CorrectnessOracle::compare(
               values(normalized), expectedNorm, {1.0e-5F, 1.0e-5F}).matches,
           "final RMSNorm matches the independent oracle");

    transformer::output::LMHead tiedHead(
        backend, 4, 6, models::TensorLayout::OutputInput, true);
    const auto logits = tiedHead.execute(normalized.view(), embeddingTable.view());
    const auto expectedLogits = validation::CorrectnessOracle::lmHead(
        expectedNorm, 2, 4, embeddingWeights, 6, true);
    expect(validation::CorrectnessOracle::compare(
               values(logits), expectedLogits, {1.0e-5F, 1.0e-5F}).matches,
           "tied vocabulary-by-hidden LM head matches the independent oracle");
}

void testArtifactToLogits() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(source);
    writeFixture(source);
    const auto imported = importer::qwen::QwenImporter{}.inspect(source);
    expect(imported.modelIO && imported.runtimeArchitecture->vocabularySize == 6 &&
               !imported.modelIO->tiedEmbeddings &&
               imported.modelIO->lmHead.layout == models::TensorLayout::OutputInput,
           "Qwen importer discovers embedding, final norm, and LM head metadata");
    const auto packing = conversion::ExpertPacker{}.pack(imported, source, packed);
    const auto manifest = models::ModelManifest::load(packed / "manifest.json");
    expect(packing.validationPassed && manifest.modelIO &&
               manifest.modelIO->lmHead.layout == models::TensorLayout::InputOutput &&
               manifest.findTensor(manifest.modelIO->tokenEmbeddingTensor)->shape ==
                   tensor::Shape{6,4},
           "artifact packing preserves model I/O and converts the separate head layout");

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
    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    auto router = std::make_shared<router::Router>(
        manifest.router.config, std::make_shared<router::CpuRouterBackend>());
    auto moeRuntime = std::make_shared<runtime::MoERuntime>(
        router, scheduler, experts, expertMappings(manifest), backend,
        std::make_shared<ExpertMlpExecutor>(backend));
    const auto architecture = models::runtime::ModelArchitecture::fromManifest(manifest);
    auto norm = std::make_shared<transformer::norm::RMSNorm>(
        backend, 4, architecture.inputNormalization.epsilon);
    auto transformer = std::make_shared<models::runtime::TransformerModelRuntime>(
        manifest, loadRuntimeTensors(manifest, store->dataPath(), backend),
        std::make_shared<transformer::attention::CpuAttention>(backend), norm, norm,
        std::make_shared<transformer::MoELayer>(moeRuntime, backend), backend);
    models::runtime::ModelRuntime model(manifest, transformer, backend);
    runtime::InferenceContext context;
    context.batchSize = 2;
    context.hiddenDimension = 4;
    const std::vector<std::uint32_t> tokenIds{1,5};
    const auto result = model.forward(context, tokenIds);

    const auto expectedEmbedding = validation::CorrectnessOracle::embedding(
        tokenIds, embeddingWeights, 6, 4);
    const auto expectedLayer = referenceLayer(expectedEmbedding);
    const auto expectedNorm = validation::CorrectnessOracle::rmsNorm(
        expectedLayer, 2, 4, finalNormWeights, 1.0e-6F);
    const auto expectedLogits = validation::CorrectnessOracle::lmHead(
        expectedNorm, 2, 4, headWeights, 6, true);
    const std::vector<std::vector<float>> actualLayers{
        values(result.transformer.layers.front().output)};
    const std::vector<std::vector<float>> expectedLayers{expectedLayer};
    const auto comparison = validation::CorrectnessOracle::compareForward(
        values(result.embeddings), expectedEmbedding, actualLayers, expectedLayers,
        values(result.normalizedHiddenStates), expectedNorm,
        values(result.logits), expectedLogits);
    expect(comparison.matches() && result.logits.shape() == tensor::Shape{2,6} &&
               result.timings.total.count() > 0 &&
               model.architecture().vocabularySize == 6,
           "complete artifact-to-logits runtime matches the independent forward oracle");
    expectThrows([&] {
        runtime::InferenceContext invalid = context;
        invalid.batchSize = 1;
        (void)model.forward(invalid, tokenIds);
    }, "model runtime rejects token/context batch disagreement");
}

void testTiedArtifactOwnership() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(source);
    writeFixture(source, true);
    const auto imported = importer::qwen::QwenImporter{}.inspect(source);
    expect(imported.modelIO && imported.modelIO->tiedEmbeddings &&
               imported.modelIO->lmHead.tensorName ==
                   imported.modelIO->tokenEmbeddingTensor &&
               imported.findTensor("lm_head.weight") == nullptr,
           "Qwen tied-weight import aliases an absent LM head to the embedding tensor");
    (void)conversion::ExpertPacker{}.pack(imported, source, packed);
    const auto manifest = models::ModelManifest::load(packed / "manifest.json");
    expect(manifest.modelIO && manifest.modelIO->tiedEmbeddings &&
               manifest.modelIO->lmHead.tensorName ==
                   manifest.modelIO->tokenEmbeddingTensor &&
               manifest.findTensor("model.lm_head") == nullptr,
           "packing a tied model preserves one shared weight tensor without duplication");
}

} // namespace

int main() {
    testForwardComponents();
    testArtifactToLogits();
    testTiedArtifactOwnership();
    if (failures != 0) {
        std::cerr << failures << " Phase 14 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 14 forward runtime tests passed\n";
    return EXIT_SUCCESS;
}
