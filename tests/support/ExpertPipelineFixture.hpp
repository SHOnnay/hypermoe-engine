#pragma once

#include "backend/CpuBackend.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/CudaRouterBackend.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

namespace hypermoe::test {

// Tiny INT8/FP32 artifact fixture. No model download and no synthetic bandwidth
// assumptions: the production loader, scheduler and executor are exercised.
class ExpertPipelineFixture {
public:
    static constexpr std::size_t expertCount = 4;
    std::filesystem::path path;
    std::shared_ptr<tensor::TensorBackend> tensors;
    std::shared_ptr<backend::ComputeBackend> compute;
    std::shared_ptr<Profiler> profiler{std::make_shared<Profiler>()};
    std::shared_ptr<storage::ExpertStore> store;
    std::shared_ptr<TransferManager> transfers;
    std::unique_ptr<MemoryManager> memory;
    std::unique_ptr<ExpertManager> experts;
    std::shared_ptr<scheduler::Scheduler> scheduler;
    std::unique_ptr<runtime::MoERuntime> runtime;
    tensor::Tensor hidden;
    tensor::Tensor routerWeights;
    std::size_t payloadBytes{};

    ExpertPipelineFixture(bool overlap, std::size_t deviceBudget,
                          std::size_t ramBudget, bool int8 = true,
                          std::shared_ptr<tensor::TensorBackend> tensorBackend = {},
                          std::shared_ptr<backend::ComputeBackend> transferBackend = {},
                          std::unique_ptr<CachePolicy> policy = {})
        : tensors(tensorBackend ? std::move(tensorBackend)
                                : std::make_shared<tensor::CpuTensorBackend>()),
          compute(transferBackend ? std::move(transferBackend)
                                  : std::make_shared<backend::CpuBackend>()) {
        static std::atomic_uint64_t sequence{};
        do {
            path = std::filesystem::temp_directory_path() /
                ("hypermoe-pipeline-" + std::to_string(sequence.fetch_add(1)));
        } while (!std::filesystem::create_directory(path));
        const std::array<float, 12> identity{
            1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1};
        std::vector<std::byte> bytes;
        if (int8) {
            for (const auto value : identity) {
                bytes.push_back(value == 0 ? std::byte{0} : std::byte{127});
            }
        } else {
            bytes.resize(sizeof(identity));
            std::memcpy(bytes.data(), identity.data(), bytes.size());
        }
        payloadBytes = bytes.size();
        std::vector<storage::ExpertBlob> blobs;
        const auto quantization = int8 ? QuantizationType::Int8 : QuantizationType::Fp32;
        for (ExpertId id = 0; id < expertCount; ++id) {
            blobs.push_back({0, id, static_cast<std::uint32_t>(quantization), bytes});
        }
        storage::ExpertStore::create(path, blobs, "{}");
        store = std::make_shared<storage::ExpertStore>(path);
        auto loader = std::make_shared<storage::DiskLoader>(store);
        transfers = std::make_shared<TransferManager>(loader, compute, 2);
        memory = std::make_unique<MemoryManager>(deviceBudget, ramBudget);
        experts = std::make_unique<ExpertManager>(
            *memory, policy ? std::move(policy) : std::make_unique<LruCachePolicy>(), transfers);
        scheduler = std::make_shared<hypermoe::scheduler::Scheduler>(
            transfers, profiler, 2, MemoryTier::Ram, ramBudget);
        models::ExpertWeightMap mappings;
        for (ExpertId id = 0; id < expertCount; ++id) {
            const auto record = store->index().find(0, id).value();
            experts->registerExpert({id, 0, payloadBytes, quantization, MemoryTier::Nvme});
            scheduler->registerExpert(0, id);
            const auto types = std::array{models::ExpertWeightType::GATE,
                                         models::ExpertWeightType::UP,
                                         models::ExpertWeightType::DOWN};
            for (std::size_t projection = 0; projection < types.size(); ++projection) {
                const auto size = payloadBytes / 3;
                mappings.add(0, id, types[projection],
                    {"fixture", {2, 2}, int8 ? tensor::DType::INT8 : tensor::DType::FP32,
                     int8 ? std::optional{tensor::quantization::QuantizedDType::INT8}
                          : std::nullopt,
                     record.offset + projection * size, size, 0, id,
                     int8 ? std::optional{tensor::quantization::QuantizationParameters{1.0F / 127.0F, 0}}
                          : std::nullopt});
            }
        }
        std::shared_ptr<router::RouterBackend> routerBackend;
        if (tensors->device().type == tensor::DeviceType::CUDA) {
            routerBackend = std::make_shared<router::CudaRouterBackend>(tensors);
        } else {
            routerBackend = std::make_shared<router::CpuRouterBackend>();
        }
        auto router = std::make_shared<router::Router>(
            router::RouterConfig{expertCount, 2, router::RoutingNormalization::Softmax, true},
            std::move(routerBackend));
        auto executor = std::make_shared<ExpertMlpExecutor>(tensors);
        runtime = std::make_unique<runtime::MoERuntime>(
            router, scheduler, *experts, std::move(mappings), tensors, executor,
            nullptr, nullptr, overlap);
        tensor::CpuTensorBackend cpu;
        auto host = cpu.allocateTensor({1, 2}, tensor::DType::FP32);
        static_cast<float*>(host.data())[0] = 1;
        static_cast<float*>(host.data())[1] = 2;
        hidden = tensors->allocateTensor(host.shape(), host.dtype());
        tensors->copyTensor(host.view(), hidden.view());
        routerWeights = tensors->allocateTensor({2, expertCount}, tensor::DType::FP32);
        select(0, 1);
    }

    void select(ExpertId first, ExpertId second) {
        tensor::CpuTensorBackend cpu;
        auto host = cpu.allocateTensor({2, expertCount}, tensor::DType::FP32);
        for (std::size_t index = 0; index < host.shape().elementCount(); ++index) {
            const auto id = index % expertCount;
            static_cast<float*>(host.data())[index] =
                id == first || id == second ? 0.0F : -10.0F;
        }
        tensors->copyTensor(host.view(), routerWeights.view());
    }

    [[nodiscard]] runtime::BatchLayerExecutionResult execute() {
        return runtime->executeBatch(0, hidden.view(), routerWeights.view());
    }

    ~ExpertPipelineFixture() {
        runtime.reset();
        if (scheduler) scheduler->shutdown();
        scheduler.reset();
        experts.reset();
        transfers.reset();
        store.reset();
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

} // namespace hypermoe::test
