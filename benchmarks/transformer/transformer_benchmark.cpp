#include "backend/CpuBackend.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
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
#include "tensor/backend/CudaTensorBackend.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/norm/RMSNorm.hpp"
#include "transformer/runtime/TransformerBlock.hpp"

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
                ("hypermoe-transformer-benchmark-" +
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
        throw std::invalid_argument("benchmark tensor data has wrong size");
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
        const std::filesystem::path reportPath =
            argc > 1 ? argv[1] : "transformer_report.json";
        TemporaryDirectory temporary;
        const std::vector<float> identity{1, 0, 0, 1};
        const std::vector<float> half{0.5F, 0, 0, 0.5F};
        const std::vector<float> twice{2, 0, 0, 2};
        const std::vector<std::vector<float>> gates{identity, half, identity};
        const std::vector<std::vector<float>> ups{identity, twice, half};
        std::vector<storage::ExpertBlob> blobs;
        for (std::uint32_t expert = 0; expert < 3; ++expert) {
            std::vector<float> packed;
            packed.insert(packed.end(), gates[expert].begin(), gates[expert].end());
            packed.insert(packed.end(), ups[expert].begin(), ups[expert].end());
            packed.insert(packed.end(), identity.begin(), identity.end());
            blobs.push_back({0, expert, 0, bytes(packed)});
        }
        storage::ExpertStore::create(
            temporary.path(), blobs, R"({"benchmark":"phase12_transformer"})");
        auto store = std::make_shared<storage::ExpertStore>(temporary.path());
        auto loader = std::make_shared<storage::DiskLoader>(store);
        auto transfers = std::make_shared<TransferManager>(
            loader, std::make_shared<backend::CpuBackend>(), 2);
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
                              4 * sizeof(float), record.layer_id,
                              record.expert_id});
            };
            add(models::ExpertWeightType::GATE, "gate", 0);
            add(models::ExpertWeightType::UP, "up", 4 * sizeof(float));
            add(models::ExpertWeightType::DOWN, "down", 8 * sizeof(float));
        }

        auto cpu = std::make_shared<tensor::CpuTensorBackend>();
        const router::RouterConfig config{
            3, 2, router::RoutingNormalization::Softmax, true};
        auto router = std::make_shared<router::Router>(
            config, std::make_shared<router::CpuRouterBackend>());
        auto executor = std::make_shared<ExpertMlpExecutor>(cpu);
        auto runtime = std::make_shared<runtime::MoERuntime>(
            router, scheduler, experts, std::move(mappings), cpu, executor);
        auto moe = std::make_shared<transformer::MoELayer>(runtime, cpu);
        auto attention =
            std::make_shared<transformer::attention::CpuAttention>(cpu);
        auto normalization =
            std::make_shared<transformer::norm::RMSNorm>(cpu, 2);
        transformer::runtime::TransformerBlock block(
            attention, normalization, moe, cpu);

        constexpr std::size_t tokenCount = 8;
        const std::vector<float> hiddenValues{
            1, 0, 0, 1, 1, 1, -1, 1,
            2, 1, 1, 2, -1, 0, 0, -1};
        const std::vector<float> routerValues{3, 2, 0, 0, 2, 3};
        const std::vector<float> normValues{1, 1};
        auto hidden = makeTensor(cpu, {tokenCount, 2}, hiddenValues);
        auto routerWeights = makeTensor(cpu, {2, 3}, routerValues);
        auto query = makeTensor(cpu, {2, 2}, identity);
        auto key = makeTensor(cpu, {2, 2}, identity);
        auto value = makeTensor(cpu, {2, 2}, identity);
        auto output = makeTensor(cpu, {2, 2}, identity);
        auto norm = makeTensor(cpu, {2}, normValues);
        const transformer::runtime::TransformerBlockWeights weights{
            {query, key, value, output}, norm, routerWeights, {}, {}};

        runtime::InferenceContext context;
        context.batchSize = tokenCount;
        context.hiddenDimension = 2;
        context.layerIndex = 0;
        (void)block.execute(context, hidden, weights);

        constexpr std::uint32_t iterations = 100;
        std::chrono::nanoseconds attentionTime{};
        std::chrono::nanoseconds normTime{};
        std::chrono::nanoseconds routingTime{};
        std::chrono::nanoseconds schedulingTime{};
        std::chrono::nanoseconds expertTime{};
        std::chrono::nanoseconds combinationTime{};
        std::chrono::nanoseconds residualTime{};
        std::chrono::nanoseconds totalTime{};
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            context.advanceLayer(0);
            const auto result = block.execute(context, hidden, weights);
            attentionTime += result.timings.attention;
            normTime += result.timings.normalization;
            routingTime += result.moe.execution.routingTime;
            schedulingTime += result.moe.execution.schedulingTime;
            expertTime += result.moe.execution.expertExecutionTime;
            combinationTime += result.moe.execution.expertCombinationTime;
            residualTime += result.timings.residual;
            totalTime += result.timings.total;
        }
        const auto divisor = static_cast<double>(iterations);
        const auto totalMs = milliseconds(totalTime) / divisor;
        const auto tokensPerSecond =
            totalMs <= 0.0 ? 0.0 : static_cast<double>(tokenCount) * 1000.0 / totalMs;
        const tensor::CudaTensorBackend cuda;
        const bool cudaAvailable = cuda.available();
        const std::string cudaReason = cudaAvailable
            ? "CUDA transformer layers are interface-ready; CUDA attention and RMSNorm are pending"
            : "CUDA/cuBLAS runtime is unavailable on this host";

        std::ofstream report(reportPath);
        report << std::fixed << std::setprecision(6)
               << "{\n"
               << "  \"benchmark\": \"phase12_transformer_pipeline\",\n"
               << "  \"fixture\": \"reduced_cpu_correctness_workload\",\n"
               << "  \"iterations\": " << iterations << ",\n"
               << "  \"tokens_per_iteration\": " << tokenCount << ",\n"
               << "  \"cpu\": {\n"
               << "    \"attention_ms\": " << milliseconds(attentionTime) / divisor << ",\n"
               << "    \"rmsnorm_ms\": " << milliseconds(normTime) / divisor << ",\n"
               << "    \"routing_ms\": " << milliseconds(routingTime) / divisor << ",\n"
               << "    \"expert_scheduling_ms\": " << milliseconds(schedulingTime) / divisor << ",\n"
               << "    \"expert_execution_ms\": " << milliseconds(expertTime) / divisor << ",\n"
               << "    \"expert_combination_ms\": " << milliseconds(combinationTime) / divisor << ",\n"
               << "    \"residual_ms\": " << milliseconds(residualTime) / divisor << ",\n"
               << "    \"transformer_block_ms\": " << totalMs << ",\n"
               << "    \"tokens_per_second\": " << tokensPerSecond << "\n"
               << "  },\n"
               << "  \"cuda\": {\n"
               << "    \"available\": " << (cudaAvailable ? "true" : "false") << ",\n"
               << "    \"executed\": false,\n"
               << "    \"reason\": \"" << cudaReason << "\"\n"
               << "  }\n"
               << "}\n";
        if (!report) throw std::runtime_error("failed to write transformer report");
        std::cout << "Transformer block CPU: " << std::fixed
                  << std::setprecision(6) << totalMs << " ms, "
                  << tokensPerSecond << " tokens/s\n"
                  << "CUDA benchmark skipped: " << cudaReason << '\n'
                  << "Report: " << reportPath << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Transformer benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
