#include "backend/CpuBackend.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
#include "experts/ExpertBatch.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "runtime/InferenceContext.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/norm/RMSNorm.hpp"
#include "transformer/runtime/TransformerBlock.hpp"
#include "validation/CorrectnessOracle.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
                ("hypermoe-phase12-" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

std::vector<std::byte> asBytes(std::span<const float> values) {
    std::vector<std::byte> bytes(values.size_bytes());
    std::memcpy(bytes.data(), values.data(), values.size_bytes());
    return bytes;
}

hypermoe::tensor::Tensor makeTensor(
    const std::shared_ptr<hypermoe::tensor::CpuTensorBackend>& backend,
    const hypermoe::tensor::Shape& shape,
    std::span<const float> values) {
    auto tensor = backend->allocateTensor(shape, hypermoe::tensor::DType::FP32);
    if (tensor.bytes() != values.size_bytes()) {
        throw std::invalid_argument("test tensor shape does not match values");
    }
    std::memcpy(tensor.data(), values.data(), values.size_bytes());
    return tensor;
}

std::span<const float> floats(const hypermoe::tensor::Tensor& tensor) {
    return {static_cast<const float*>(tensor.data()),
            tensor.shape().elementCount()};
}

void testExpertBatchAndContext() {
    using namespace hypermoe;
    ExpertBatch batch{3, 7, {0, 2}, {0.75F, 0.25F}};
    batch.validate(3);
    expect(batch.size() == 2, "expert batch reports grouped token count");
    expectThrows([] {
        ExpertBatch invalid{0, 1, {0, 0}, {0.5F, 0.5F}};
        invalid.validate(1);
    }, "expert batch rejects duplicate token assignments");
    expectThrows([] {
        ExpertBatch invalid{0, 1, {2}, {1.0F}};
        invalid.validate(2);
    }, "expert batch rejects out-of-range token assignments");

    runtime::InferenceContext context;
    context.batchSize = 2;
    context.sequencePosition = 11;
    context.hiddenDimension = 4;
    context.layerIndex = 3;
    context.validate();
    router::BatchRouterDecision routing;
    routing.layerId = 3;
    routing.tokens = {{3, {0}, {1.0F}}, {3, {1}, {1.0F}}};
    routing.expertGroups = {{0, {0}, {1.0F}}, {1, {1}, {1.0F}}};
    runtime::ExecutionMetadata metadata;
    metadata.expertAssignments = 2;
    context.recordRouting(routing, metadata);
    expect(context.routingDecisions.size() == 2 &&
               context.execution.expertAssignments == 2,
           "inference context records routing and execution metadata");
    context.advanceLayer(4);
    expect(context.layerIndex == 4 && context.routingDecisions.empty() &&
               context.execution.expertAssignments == 0,
           "advancing inference context clears layer-local state");
    context.batchSize = 0;
    expectThrows([&] { context.validate(); },
                 "inference context rejects zero-sized batches");
}

void testAttentionAndNormalization() {
    using namespace hypermoe;
    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    const std::vector<float> hiddenValues{1, 0, 0, 1};
    const std::vector<float> identity{1, 0, 0, 1};
    auto hidden = makeTensor(backend, {2, 2}, hiddenValues);
    auto query = makeTensor(backend, {2, 2}, identity);
    auto key = makeTensor(backend, {2, 2}, identity);
    auto value = makeTensor(backend, {2, 2}, identity);
    auto outputProjection = makeTensor(backend, {2, 2}, identity);
    transformer::attention::CpuAttention attention(backend);
    const auto result = attention.execute(
        hidden, {query, key, value, outputProjection});
    const auto diagonal = std::exp(1.0F / std::sqrt(2.0F));
    const auto selected = diagonal / (diagonal + 1.0F);
    const auto other = 1.0F / (diagonal + 1.0F);
    const std::vector<float> expected{selected, other, other, selected};
    expect(validation::CorrectnessOracle::compare(
               floats(result.output), expected,
               validation::CorrectnessOracle::toleranceFor(
                   tensor::DType::FP32)).matches,
           "CPU attention performs QKV projection, scaled softmax, and output projection");
    auto badProjection = makeTensor(backend, {1, 2}, std::vector<float>{1, 1});
    expectThrows([&] {
        (void)attention.execute(
            hidden, {badProjection, key, value, outputProjection});
    }, "CPU attention rejects incompatible projection shapes");

    auto normInput = makeTensor(backend, {2, 2},
                                std::vector<float>{3, 4, 0, 2});
    auto normWeight = makeTensor(backend, {2}, std::vector<float>{1, 2});
    transformer::norm::RMSNorm normalization(backend, 2, 1.0e-6F);
    const auto normalized = normalization.execute(normInput, normWeight);
    const auto firstInverse = static_cast<float>(
        1.0 / std::sqrt(12.5 + 1.0e-6));
    const auto secondInverse = static_cast<float>(
        1.0 / std::sqrt(2.0 + 1.0e-6));
    const std::vector<float> normalizedExpected{
        3.0F * firstInverse, 8.0F * firstInverse,
        0.0F, 4.0F * secondInverse};
    expect(validation::CorrectnessOracle::compare(
               floats(normalized), normalizedExpected,
               validation::CorrectnessOracle::toleranceFor(
                   tensor::DType::FP32)).matches,
           "RMSNorm matches the independent reference values for multiple tokens");
    expectThrows([&] {
        transformer::norm::RMSNorm invalid(backend, 2, 0.0F);
        (void)invalid;
    }, "RMSNorm rejects a non-positive epsilon");
}

void testBatchedMoEAndTransformerBlock() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const std::vector<std::vector<float>> gates{
        {1, 0, 0, 1}, {0.5F, 0, 0, 0.5F}, {1, 0, 0, 1}};
    const std::vector<std::vector<float>> ups{
        {1, 0, 0, 1}, {2, 0, 0, 2}, {0.5F, 0, 0, 0.5F}};
    const std::vector<std::vector<float>> downs{
        {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}};
    std::vector<storage::ExpertBlob> blobs;
    for (std::uint32_t expert = 0; expert < 3; ++expert) {
        std::vector<float> packed;
        packed.insert(packed.end(), gates[expert].begin(), gates[expert].end());
        packed.insert(packed.end(), ups[expert].begin(), ups[expert].end());
        packed.insert(packed.end(), downs[expert].begin(), downs[expert].end());
        blobs.push_back({0, expert, 0, asBytes(packed)});
    }
    storage::ExpertStore::create(temporary.path(), blobs,
                                 R"({"phase":12,"dtype":"FP32"})");
    auto store = std::make_shared<storage::ExpertStore>(temporary.path());
    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto compute = std::make_shared<backend::CpuBackend>();
    auto transfers = std::make_shared<TransferManager>(loader, compute, 2);
    MemoryManager memory(4096, 4096);
    ExpertManager experts(memory, std::make_unique<LruCachePolicy>(), transfers);
    auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, nullptr, 2);
    models::ExpertWeightMap mappings;
    for (const auto& record : store->index().records()) {
        experts.registerExpert({record.expert_id, record.layer_id,
                                static_cast<std::size_t>(record.size),
                                QuantizationType::Fp32, MemoryTier::Nvme});
        scheduler->registerExpert(record.layer_id, record.expert_id);
        const auto add = [&](models::ExpertWeightType type,
                             std::string name,
                             std::uint64_t relativeOffset) {
            mappings.add(record.layer_id, record.expert_id, type,
                         {std::move(name), {2, 2}, tensor::DType::FP32,
                          std::nullopt, record.offset + relativeOffset,
                          4 * sizeof(float), record.layer_id, record.expert_id});
        };
        add(models::ExpertWeightType::GATE, "gate", 0);
        add(models::ExpertWeightType::UP, "up", 4 * sizeof(float));
        add(models::ExpertWeightType::DOWN, "down", 8 * sizeof(float));
    }

    auto tensors = std::make_shared<tensor::CpuTensorBackend>();
    const router::RouterConfig routerConfig{
        3, 2, router::RoutingNormalization::Softmax, true};
    auto runtimeRouter = std::make_shared<router::Router>(
        routerConfig, std::make_shared<router::CpuRouterBackend>());
    auto executor = std::make_shared<ExpertMlpExecutor>(tensors);
    auto runtime = std::make_shared<runtime::MoERuntime>(
        runtimeRouter, scheduler, experts, std::move(mappings), tensors, executor);
    auto moeLayer = std::make_shared<transformer::MoELayer>(runtime, tensors);
    const std::vector<float> hiddenValues{1, 0, 0, 1, 1, 1};
    const std::vector<float> routerValues{3, 2, 0, 0, 2, 3};
    auto hidden = makeTensor(tensors, {3, 2}, hiddenValues);
    auto routerWeights = makeTensor(tensors, {2, 3}, routerValues);

    const auto batch = runtime->executeBatch(0, hidden, routerWeights);
    expect(batch.routing.tokens.size() == 3 && batch.expertBatches.size() == 3 &&
               batch.execution.expertAssignments == 6 &&
               batch.execution.uniqueExperts == 3,
           "batched MoE groups top-k token assignments by unique expert");
    expect(batch.expertBatches[0].tokenIndices == std::vector<std::size_t>({0, 2}) &&
               batch.expertBatches[1].tokenIndices ==
                   std::vector<std::size_t>({0, 1, 2}) &&
               batch.expertBatches[2].tokenIndices == std::vector<std::size_t>({1}),
           "expert grouping preserves deterministic token-to-expert assignments");

    router::BatchRouterDecision oracleRouting;
    oracleRouting.layerId = 0;
    for (std::size_t token = 0; token < 3; ++token) {
        oracleRouting.tokens.push_back(validation::CorrectnessOracle::route(
            0, std::span<const float>(hiddenValues).subspan(token * 2, 2),
            routerValues, routerConfig));
    }
    oracleRouting.expertGroups = batch.routing.expertGroups;
    std::vector<validation::RoutedExpertOutput> expectedExperts;
    std::vector<validation::RoutedExpertOutput> actualExperts;
    for (std::size_t index = 0; index < batch.expertBatches.size(); ++index) {
        const auto& group = batch.expertBatches[index];
        std::vector<float> groupedInput;
        for (const auto token : group.tokenIndices) {
            groupedInput.insert(groupedInput.end(),
                                hiddenValues.begin() +
                                    static_cast<std::ptrdiff_t>(token * 2),
                                hiddenValues.begin() +
                                    static_cast<std::ptrdiff_t>(token * 2 + 2));
        }
        const auto expert = static_cast<std::size_t>(group.expertId);
        expectedExperts.push_back({
            group.expertId, group.tokenIndices, group.routingWeights,
            validation::CorrectnessOracle::expertMlp(
                groupedInput, group.size(), 2, gates[expert], ups[expert],
                downs[expert], 2)});
        actualExperts.push_back({group.expertId, group.tokenIndices,
                                 group.routingWeights,
                                 std::vector<float>(floats(batch.expertOutputs[index]).begin(),
                                                    floats(batch.expertOutputs[index]).end())});
    }
    const auto combinedReference = validation::CorrectnessOracle::combineExperts(
        3, 2, expectedExperts);
    const auto comparison = validation::CorrectnessOracle::compareExpertCombination(
        batch.routing, oracleRouting, actualExperts, expectedExperts,
        floats(batch.output), combinedReference.output, tensor::DType::FP32);
    expect(comparison.matches(),
           "independent oracle validates routing normalization, every expert, and weighted sum");

    const std::vector<float> identity{1, 0, 0, 1};
    const std::vector<float> normValues{1, 1};
    auto query = makeTensor(tensors, {2, 2}, identity);
    auto key = makeTensor(tensors, {2, 2}, identity);
    auto value = makeTensor(tensors, {2, 2}, identity);
    auto attentionOutput = makeTensor(tensors, {2, 2}, identity);
    auto normWeight = makeTensor(tensors, {2}, normValues);
    auto attention = std::make_shared<transformer::attention::CpuAttention>(tensors);
    auto normalization = std::make_shared<transformer::norm::RMSNorm>(tensors, 2);
    transformer::runtime::TransformerBlock block(
        attention, normalization, moeLayer, tensors);
    runtime::InferenceContext context;
    context.batchSize = 3;
    context.sequencePosition = 17;
    context.hiddenDimension = 2;
    context.layerIndex = 0;
    const auto blockResult = block.execute(
        context, hidden,
        {{query, key, value, attentionOutput}, normWeight, routerWeights, {}, {}});
    std::vector<float> residualExpected(6);
    const auto attentionValues = floats(blockResult.attention.output);
    const auto moeValues = floats(blockResult.moe.output);
    for (std::size_t index = 0; index < residualExpected.size(); ++index) {
        residualExpected[index] = attentionValues[index] + moeValues[index];
    }
    expect(validation::CorrectnessOracle::compare(
               floats(blockResult.output), residualExpected,
               validation::CorrectnessOracle::toleranceFor(
                   tensor::DType::FP32)).matches &&
               context.routingDecisions.size() == 3 &&
               context.execution.expertAssignments == 6 &&
               blockResult.timings.total.count() > 0,
           "transformer block composes attention, RMSNorm, grouped MoE, residual, and context");
}

} // namespace

int main() {
    testExpertBatchAndContext();
    testAttentionAndNormalization();
    testBatchedMoEAndTransformerBlock();
    if (failures != 0) {
        std::cerr << failures << " Phase 12 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 12 transformer pipeline tests passed\n";
    return EXIT_SUCCESS;
}
