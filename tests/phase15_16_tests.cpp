#include "backend/CpuBackend.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
#include "generation/Generator.hpp"
#include "generation/LogitsProcessor.hpp"
#include "generation/Sampler.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "models/runtime/ModelRuntime.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "runtime/cache/KVCacheManager.hpp"
#include "runtime/generation/Decoder.hpp"
#include "runtime/generation/GenerationModel.hpp"
#include "runtime/generation/InferenceSession.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tokenizer/qwen/QwenTokenizerAdapter.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/norm/RMSNorm.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

template <typename Exception = std::exception, typename Function>
void expectThrows(Function&& function, std::string_view message) {
    try {
        function();
        expect(false, message);
    } catch (const Exception&) {
        expect(true, message);
    } catch (...) {
        expect(false, message);
    }
}

class DeterministicGenerationModel final
    : public hypermoe::runtime::generation::GenerationModel {
public:
    DeterministicGenerationModel()
        : backend_(std::make_shared<hypermoe::tensor::CpuTensorBackend>()) {}

    [[nodiscard]] std::size_t vocabularySize() const noexcept override { return 8; }
    [[nodiscard]] std::size_t hiddenDimension() const noexcept override { return 4; }
    [[nodiscard]] std::size_t layerCount() const noexcept override { return 2; }
    [[nodiscard]] std::size_t keyValueHeads() const noexcept override { return 1; }
    [[nodiscard]] std::size_t headDimension() const noexcept override { return 2; }
    [[nodiscard]] hypermoe::tensor::Device device() const noexcept override {
        return hypermoe::tensor::Device::cpu();
    }

    [[nodiscard]] hypermoe::runtime::generation::ForwardPass forward(
        hypermoe::runtime::InferenceContext& context,
        std::span<const std::uint32_t> tokenIds,
        hypermoe::runtime::cache::KVCacheBase& cache) override {
        if (tokenIds.empty() || context.batchSize != tokenIds.size() ||
            context.hiddenDimension != hiddenDimension()) {
            throw std::invalid_argument("fixture forward context is incompatible");
        }
        auto hidden = backend_->allocateTensor(
            {tokenIds.size(), hiddenDimension()}, hypermoe::tensor::DType::FP32);
        auto logits = backend_->allocateTensor(
            {tokenIds.size(), vocabularySize()}, hypermoe::tensor::DType::FP32);
        auto* hiddenValues = static_cast<float*>(hidden.data());
        auto* logitValues = static_cast<float*>(logits.data());
        for (std::size_t token = 0; token < tokenIds.size(); ++token) {
            if (tokenIds[token] >= vocabularySize()) {
                throw std::out_of_range("fixture token exceeds vocabulary");
            }
            for (std::size_t feature = 0; feature < hiddenDimension(); ++feature) {
                hiddenValues[token * hiddenDimension() + feature] =
                    static_cast<float>(tokenIds[token] + feature);
            }
            for (std::size_t vocabulary = 0; vocabulary < vocabularySize(); ++vocabulary) {
                logitValues[token * vocabularySize() + vocabulary] = -8.0F;
            }
            const auto next = static_cast<std::size_t>(
                (tokenIds[token] + 1U) % vocabularySize());
            logitValues[token * vocabularySize() + next] = 8.0F;
        }
        for (std::size_t layer = 0; layer < layerCount(); ++layer) {
            auto keys = backend_->allocateTensor(
                {tokenIds.size(), keyValueHeads(), headDimension()},
                hypermoe::tensor::DType::FP32);
            auto values = backend_->allocateTensor(
                keys.shape(), hypermoe::tensor::DType::FP32);
            auto* keyValues = static_cast<float*>(keys.data());
            auto* valueValues = static_cast<float*>(values.data());
            for (std::size_t index = 0; index < keys.shape().elementCount(); ++index) {
                keyValues[index] = static_cast<float>(layer + index);
                valueValues[index] = static_cast<float>(layer + index + 1U);
            }
            cache.append(layer, context.sequencePosition, keys.view(), values.view());
        }
        std::vector<hypermoe::router::RouterDecision> routing(
            tokenIds.size(), {1, {0}, {1.0F}});
        return {std::move(hidden), std::move(logits), 1, std::move(routing)};
    }

private:
    std::shared_ptr<hypermoe::tensor::CpuTensorBackend> backend_;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        std::error_code probe;
        do {
            path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-phase15-16-" + std::to_string(
                    sequence.fetch_add(1, std::memory_order_relaxed)));
            probe.clear();
        } while (std::filesystem::exists(path_, probe) || probe);
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

std::vector<std::byte> asBytes(std::span<const float> values) {
    std::vector<std::byte> result(values.size_bytes());
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

hypermoe::tensor::Tensor makeTensor(
    const std::shared_ptr<hypermoe::tensor::CpuTensorBackend>& backend,
    const hypermoe::tensor::Shape& shape,
    std::span<const float> values) {
    auto result = backend->allocateTensor(shape, hypermoe::tensor::DType::FP32);
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

std::shared_ptr<hypermoe::tokenizer::Tokenizer> makeTokenizer() {
    using hypermoe::tokenizer::TokenId;
    return std::make_shared<hypermoe::tokenizer::qwen::QwenTokenizerAdapter>(
        8,
        [](std::string_view text) {
            std::vector<TokenId> result;
            result.reserve(text.size());
            for (const char character : text) {
                if (character < '0' || character > '7') {
                    throw std::invalid_argument("fixture tokenizer accepts digits 0-7");
                }
                result.push_back(static_cast<TokenId>(character - '0'));
            }
            return result;
        },
        [](std::span<const TokenId> tokens) {
            std::string result;
            result.reserve(tokens.size());
            for (const auto token : tokens) {
                result.push_back(static_cast<char>('0' + token));
            }
            return result;
        });
}

void testTokenizerAdapter() {
    const auto tokenizer = makeTokenizer();
    const auto tokens = tokenizer->encode("127");
    expect(tokens == std::vector<std::uint32_t>({1,2,7}) &&
               tokenizer->decode(tokens) == "127" &&
               tokenizer->vocabularySize() == 8,
           "Qwen tokenizer adapter validates and delegates encode/decode");
    expectThrows<std::out_of_range>(
        [&] {
            const std::vector<std::uint32_t> invalid{8};
            (void)tokenizer->decode(invalid);
        }, "tokenizer rejects IDs outside declared vocabulary");
}

void testSampling() {
    using namespace hypermoe::generation;
    const std::vector<float> logits{1.0F, 5.0F, 5.0F, 2.0F};
    Sampler greedy;
    expect(greedy.sample(logits) == 1U,
           "greedy sampling resolves equal logits by lowest token ID");

    SamplingConfig filtered;
    filtered.strategy = SamplingStrategy::Stochastic;
    filtered.temperature = 0.7F;
    filtered.topK = 2;
    filtered.topP = 0.6F;
    filtered.seed = 42;
    const auto probabilities = LogitsProcessor::probabilities(logits, filtered);
    std::size_t nonzero{};
    double sum{};
    for (const auto probability : probabilities) {
        if (probability > 0.0) ++nonzero;
        sum += probability;
    }
    expect(nonzero >= 1 && nonzero <= 2 && std::abs(sum - 1.0) < 1.0e-12,
           "temperature, top-k, and top-p produce a normalized filtered distribution");
    Sampler first(filtered);
    Sampler second(filtered);
    bool deterministic = true;
    for (std::size_t index = 0; index < 16; ++index) {
        deterministic = deterministic && first.sample(logits) == second.sample(logits);
    }
    expect(deterministic, "seeded sampling is deterministic across sampler instances");
    filtered.topP = 0.0F;
    expectThrows<std::invalid_argument>(
        [&] { (void)LogitsProcessor::probabilities(logits, filtered); },
        "sampling rejects an invalid top-p threshold");
}

void testKVCacheManagerAndDecoder() {
    using namespace hypermoe;
    auto model = std::make_shared<DeterministicGenerationModel>();
    auto manager = std::make_shared<runtime::cache::KVCacheManager>(2, 8, 1, 2, 384);
    expect(manager->bytesPerSession() == 384,
           "KV cache manager calculates a bounded maximum reservation");
    {
        runtime::generation::InferenceSession session(model, manager, 3);
        expectThrows<std::runtime_error>(
            [&] { (void)manager->allocateSession(); },
            "KV cache manager rejects sessions beyond its memory limit");
        runtime::generation::Decoder decoder;
        const std::vector<std::uint32_t> prompt{1,2};
        decoder.prefill(session, prompt);
        expect(session.generationState().currentTokenPosition() == 2 &&
                   session.kvCache().tokenCount(0) == 2 &&
                   session.kvCache().tokenCount(1) == 2 &&
                   session.forwardState().attention().cachedTokenCount == 2,
               "prefill populates every layer and advances the sequence position");
        decoder.decode(session, 3);
        const auto stats = manager->stats();
        expect(session.generationState().currentTokenPosition() == 3 &&
                   session.kvCache().tokenCount(0) == 3 &&
                   stats.activeSessions == 1 && stats.committedBytes == 144 &&
                   stats.reservedBytes == 384,
               "single-token decode grows the bounded cache with exact accounting");
    }
    const auto released = manager->stats();
    expect(released.activeSessions == 0 && released.committedBytes == 0 &&
               released.reservedBytes == 0 && released.peakCommittedBytes == 144,
           "inference session destruction releases its KV cache allocation");
}

void testGenerationIntegration() {
    using namespace hypermoe;
    auto model = std::make_shared<DeterministicGenerationModel>();
    auto manager = std::make_shared<runtime::cache::KVCacheManager>(2, 8, 1, 2, 768);
    generation::Generator generator(makeTokenizer(), model, manager);
    generation::GenerationConfig config;
    config.maximumNewTokens = 3;
    const auto result = generator.generate("12", config);
    expect(result.promptTokens == std::vector<std::uint32_t>({1,2}) &&
               result.generatedTokens == std::vector<std::uint32_t>({3,4,5}) &&
               result.text == "345" && result.metrics.decodeSteps == 2 &&
               result.stopReason ==
                   runtime::generation::GenerationStopReason::MaximumTokens &&
               result.metrics.peakKVCacheBytes == 192,
           "generator performs tokenize, prefill, incremental decode, and detokenize");
    config.maximumNewTokens = 5;
    config.stopTokenIds = {4};
    const auto stopped = generator.generate("12", config);
    expect(stopped.generatedTokens == std::vector<std::uint32_t>({3,4}) &&
               stopped.stopReason ==
                   runtime::generation::GenerationStopReason::StopToken &&
               manager->stats().activeSessions == 0,
           "generation stops on configured token and releases session memory");
    config.maximumNewTokens = 8;
    expectThrows<std::invalid_argument>(
        [&] { (void)generator.generate("12", config); },
        "generation rejects prompt/output combinations beyond cache capacity");
}

void testRealModelRuntimeGenerationAdapter() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const std::vector<float> identity{1,0,0,1};
    std::vector<float> packed;
    for (std::size_t projection = 0; projection < 3; ++projection) {
        packed.insert(packed.end(), identity.begin(), identity.end());
    }
    const std::vector<storage::ExpertBlob> blobs{
        {0, 0, 0, asBytes(packed)}};
    storage::ExpertStore::create(
        temporary.path(), blobs, R"({"fixture":"phase15_real_model"})");
    auto store = std::make_shared<storage::ExpertStore>(temporary.path());
    const auto record = store->index().records().front();

    models::ModelManifest manifest;
    manifest.modelName = "Phase 15 real runtime fixture";
    manifest.architecture = models::ModelArchitecture::QWEN_MOE;
    manifest.sourceArchitecture = "Qwen3MoeForCausalLM";
    manifest.config = {manifest.modelName, 1, 1, 2, 2,
                       {true, true, true, true, false, false}};
    manifest.router.config = {1, 1, router::RoutingNormalization::Softmax, true};
    manifest.router.layout = models::TensorLayout::InputOutput;
    models::runtime::ModelArchitecture architecture;
    architecture.layerCount = 1;
    architecture.hiddenDimension = 2;
    architecture.attentionHeads = 1;
    architecture.keyValueHeads = 1;
    architecture.headDimension = 2;
    architecture.projectionHeadDimension = 2;
    architecture.expertCount = 1;
    architecture.topK = 1;
    architecture.vocabularySize = 4;
    manifest.runtimeArchitecture = architecture;

    models::ExpertWeightMap expertWeights;
    models::ManifestExpertMapping expert;
    expert.layerId = 0;
    expert.expertId = 0;
    const auto projectionBytes = static_cast<std::uint64_t>(4 * sizeof(float));
    const auto addExpertProjection = [&](std::string name,
                                         std::uint64_t relative,
                                         models::ExpertWeightType type,
                                         models::ProjectionLocation& location) {
        const auto offset = record.offset + relative;
        manifest.tensors.push_back(
            {name, "experts.bin", offset, projectionBytes,
             tensor::DType::FP32, {2,2}});
        location = {name, offset, projectionBytes, {2,2},
                    models::TensorLayout::InputOutput};
        expertWeights.add(0, 0, type,
            {name, {2,2}, tensor::DType::FP32, std::nullopt,
             offset, projectionBytes, 0, 0});
    };
    addExpertProjection("expert.gate", 0, models::ExpertWeightType::GATE,
                        expert.gate);
    addExpertProjection("expert.up", projectionBytes,
                        models::ExpertWeightType::UP, expert.up);
    addExpertProjection("expert.down", 2U * projectionBytes,
                        models::ExpertWeightType::DOWN, expert.down);
    manifest.experts.push_back(expert);

    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    models::runtime::RuntimeTensorMap tensors;
    const auto addTensor = [&](std::string name, const tensor::Shape& shape,
                               std::span<const float> data) {
        manifest.tensors.push_back(
            {name, "experts.bin", 0,
             static_cast<std::uint64_t>(data.size_bytes()),
             tensor::DType::FP32, shape});
        tensors.add(name, makeTensor(backend, shape, data));
        return name;
    };
    const std::vector<float> norm{1,1};
    const std::vector<float> routerWeights{0,0};
    models::ManifestLayerMapping layer;
    layer.layerId = 0;
    layer.queryProjection = {addTensor("layer.q", {2,2}, identity),
                             models::TensorLayout::InputOutput};
    layer.keyProjection = {addTensor("layer.k", {2,2}, identity),
                           models::TensorLayout::InputOutput};
    layer.valueProjection = {addTensor("layer.v", {2,2}, identity),
                             models::TensorLayout::InputOutput};
    layer.outputProjection = {addTensor("layer.o", {2,2}, identity),
                              models::TensorLayout::InputOutput};
    layer.inputNormTensor = addTensor("layer.input_norm", {2}, norm);
    layer.postAttentionNormTensor = addTensor("layer.post_norm", {2}, norm);
    layer.routerTensor = addTensor("layer.router", {2,1}, routerWeights);
    manifest.router.tensors.push_back({0, layer.routerTensor});
    manifest.layers.push_back(layer);
    const std::vector<float> embedding{1,0, 0,1, 1,1, -1,1};
    const std::vector<float> head{1,0,1,-1, 0,1,1,1};
    const auto embeddingName = addTensor("model.embedding", {4,2}, embedding);
    const auto normName = addTensor("model.final_norm", {2}, norm);
    const auto headName = addTensor("model.lm_head", {2,4}, head);
    manifest.modelIO = models::ManifestModelIO{
        embeddingName, normName,
        {headName, models::TensorLayout::InputOutput}, false};
    manifest.validate();

    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto transfers = std::make_shared<TransferManager>(
        loader, std::make_shared<backend::CpuBackend>(), 1);
    MemoryManager memory(1U << 20U, 1U << 20U);
    ExpertManager expertManager(
        memory, std::make_unique<LruCachePolicy>(), transfers);
    expertManager.registerExpert(
        {0, 0, static_cast<std::size_t>(record.size),
         QuantizationType::Fp32, MemoryTier::Nvme});
    auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 1);
    scheduler->registerExpert(0, 0);
    auto router = std::make_shared<router::Router>(
        manifest.router.config, std::make_shared<router::CpuRouterBackend>());
    auto moe = std::make_shared<runtime::MoERuntime>(
        router, scheduler, expertManager, std::move(expertWeights), backend,
        std::make_shared<ExpertMlpExecutor>(backend));
    auto normalization = std::make_shared<transformer::norm::RMSNorm>(
        backend, 2, architecture.inputNormalization.epsilon);
    auto transformerRuntime =
        std::make_shared<models::runtime::TransformerModelRuntime>(
            manifest, std::move(tensors),
            std::make_shared<transformer::attention::CpuAttention>(backend),
            normalization, normalization,
            std::make_shared<transformer::MoELayer>(moe, backend), backend);
    auto modelRuntime = std::make_shared<models::runtime::ModelRuntime>(
        manifest, transformerRuntime, backend);
    auto generationModel =
        std::make_shared<runtime::generation::ModelRuntimeGenerationModel>(
            modelRuntime);
    runtime::cache::KVCache sizing(1, 4, 1, 2);
    auto manager = std::make_shared<runtime::cache::KVCacheManager>(
        1, 4, 1, 2, sizing.maximumMemoryUsageBytes());
    auto tokenizer =
        std::make_shared<tokenizer::qwen::QwenTokenizerAdapter>(
            4,
            [](std::string_view text) {
                std::vector<std::uint32_t> ids;
                for (const auto character : text) {
                    if (character < '0' || character > '3') {
                        throw std::invalid_argument("fixture token is invalid");
                    }
                    ids.push_back(static_cast<std::uint32_t>(character - '0'));
                }
                return ids;
            },
            [](std::span<const std::uint32_t> ids) {
                std::string text;
                for (const auto id : ids) text.push_back(static_cast<char>('0' + id));
                return text;
            });
    generation::Generator generator(tokenizer, generationModel, manager);
    generation::GenerationConfig config;
    config.maximumNewTokens = 2;
    const auto result = generator.generate("1", config);
    bool tokensInRange = result.generatedTokens.size() == 2;
    for (const auto token : result.generatedTokens) tokensInRange &= token < 4;
    expect(tokensInRange && result.text.size() == 2 &&
               result.metrics.peakKVCacheBytes > 0 &&
               manager->stats().activeSessions == 0,
           "real ModelRuntime adapter performs cached prefill and incremental decode");
    scheduler->shutdown();
    transfers->shutdown();
}

} // namespace

int main() {
    testTokenizerAdapter();
    testSampling();
    testKVCacheManagerAndDecoder();
    testGenerationIntegration();
    testRealModelRuntimeGenerationAdapter();
    if (failures != 0) {
        std::cerr << failures << " Phase 15/16 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 15/16 generation tests passed\n";
    return EXIT_SUCCESS;
}
#include <atomic>
#include <cstring>
#include <filesystem>
