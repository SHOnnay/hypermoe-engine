#include "models/runtime/PackedModelRuntime.hpp"

#include "backend/CpuBackend.hpp"
#include "backend/CudaBackend.hpp"
#include "cache/LRUPolicy.hpp"
#include "core/runtime/MoERuntime.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/cache_policy.hpp"
#include "importer/SafeTensorShardManager.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "prediction/ExpertPredictor.hpp"
#include "prediction/TransitionDatabase.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/CudaRouterBackend.hpp"
#include "router/Router.hpp"
#include "runtime/cache/CudaKVCache.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "storage/MappedFile.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"
#include "tensor/precision/DTypeConverter.hpp"
#include "transformer/MoELayer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/attention/CudaAttention.hpp"
#include "transformer/norm/RMSNorm.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypermoe::models::runtime {
namespace {

QuantizationType quantization(std::uint32_t value) {
    const auto result = static_cast<QuantizationType>(value);
    if (!isValid(result)) {
        throw std::invalid_argument("packed expert has an unsupported storage dtype");
    }
    return result;
}

models::ExpertWeightMap makeExpertMappings(const models::ModelManifest& manifest) {
    models::ExpertWeightMap result;
    for (const auto& expert : manifest.experts) {
        const auto add = [&](models::ExpertWeightType type,
                             const models::ProjectionLocation& projection) {
            const auto* tensor = manifest.findTensor(projection.tensorName);
            if (!tensor) {
                throw std::invalid_argument("packed expert projection tensor is missing");
            }
            result.add(expert.layerId, expert.expertId, type,
                       {tensor->name, projection.shape, tensor->dtype, std::nullopt,
                        projection.offset, projection.size,
                        expert.layerId, expert.expertId});
        };
        add(models::ExpertWeightType::GATE, expert.gate);
        add(models::ExpertWeightType::UP, expert.up);
        add(models::ExpertWeightType::DOWN, expert.down);
    }
    return result;
}

std::set<std::string> staticTensorNames(const models::ModelManifest& manifest) {
    std::set<std::string> result;
    for (const auto& layer : manifest.layers) {
        result.insert(layer.queryProjection.tensorName);
        result.insert(layer.keyProjection.tensorName);
        result.insert(layer.valueProjection.tensorName);
        result.insert(layer.outputProjection.tensorName);
        if (!layer.queryNormTensor.empty()) result.insert(layer.queryNormTensor);
        if (!layer.keyNormTensor.empty()) result.insert(layer.keyNormTensor);
        result.insert(layer.inputNormTensor);
        result.insert(layer.postAttentionNormTensor);
        result.insert(layer.routerTensor);
    }
    if (!manifest.modelIO) {
        throw std::invalid_argument("packed artifact has no model I/O mappings");
    }
    result.insert(manifest.modelIO->tokenEmbeddingTensor);
    result.insert(manifest.modelIO->finalNormTensor);
    result.insert(manifest.modelIO->lmHead.tensorName);
    return result;
}

void recordExecution(Profiler& profiler,
                     const ModelForwardResult& result,
                     const MemorySnapshot& memory,
                     tensor::Device device,
                     std::size_t staticBytes) {
    profiler.recordToken(result.embeddings.shape().dimensions()[0]);
    for (const auto& layer : result.transformer.layers) {
        for (std::size_t index = 0; index < layer.execution.expertCacheHits; ++index) {
            profiler.recordExpertRequest(true);
        }
        for (std::size_t index = 0; index < layer.execution.expertCacheMisses; ++index) {
            profiler.recordExpertRequest(false);
        }
        if (layer.execution.expertTransferBytes != 0) {
            profiler.recordNvmeRead(layer.execution.expertTransferBytes);
            if (device.type == tensor::DeviceType::CUDA) {
                profiler.recordRamToVram(layer.execution.expertTransferBytes);
            }
        }
    }
    const auto staticVram = device.type == tensor::DeviceType::CUDA ? staticBytes : 0;
    const auto staticRam = device.type == tensor::DeviceType::CPU ? staticBytes : 0;
    profiler.observeMemory(memory.vram.usedBytes + staticVram,
                           memory.ram.usedBytes + staticRam);
    if (device.type == tensor::DeviceType::CUDA) {
        profiler.observeGpuMemory(memory.vram.usedBytes + staticBytes);
    }
}

} // namespace

struct PackedModelRuntime::Impl {
    models::ModelManifest manifest;
    PackedRuntimeConfiguration configuration;
    std::shared_ptr<Profiler> profiler;
    std::shared_ptr<tensor::TensorBackend> tensors;
    std::shared_ptr<backend::ComputeBackend> transferBackend;
    std::shared_ptr<storage::ExpertStore> store;
    std::shared_ptr<storage::DiskLoader> loader;
    std::shared_ptr<TransferManager> transfers;
    std::unique_ptr<MemoryManager> memory;
    std::unique_ptr<ExpertManager> experts;
    std::shared_ptr<scheduler::Scheduler> scheduler;
    std::shared_ptr<prediction::ExpertHistory> history;
    std::shared_ptr<prediction::TransitionDatabase> transitions;
    std::shared_ptr<prediction::ExpertPredictor> predictor;
    std::shared_ptr<router::Router> router;
    std::shared_ptr<ExpertMlpExecutor> expertExecutor;
    std::shared_ptr<hypermoe::runtime::MoERuntime> moeRuntime;
    std::shared_ptr<transformer::MoELayer> moe;
    std::shared_ptr<transformer::attention::Attention> attention;
    std::shared_ptr<transformer::norm::RMSNorm> inputNorm;
    std::shared_ptr<transformer::norm::RMSNorm> postAttentionNorm;
    std::shared_ptr<TransformerModelRuntime> transformer;
    std::shared_ptr<ModelRuntime> model;
    std::size_t staticStorageBytes{};
    std::size_t staticExecutionBytes{};
};

void PackedRuntimeConfiguration::validate() const {
    if (expertDeviceBudgetBytes == 0 || expertRamBudgetBytes == 0 ||
        transferWorkers == 0 || schedulerWorkers == 0 || device.ordinal < 0 ||
        (device.type == tensor::DeviceType::CPU && device.ordinal != 0)) {
        throw std::invalid_argument("packed runtime configuration is invalid");
    }
}

PackedModelRuntime::PackedModelRuntime(
    const std::filesystem::path& artifact,
    PackedRuntimeConfiguration configuration)
    : impl_(std::make_unique<Impl>()) {
    configuration.validate();
    impl_->configuration = configuration;
    impl_->manifest = models::ModelManifest::load(artifact / "manifest.json");
    if (!impl_->manifest.runtimeArchitecture ||
        impl_->manifest.layers.size() !=
            impl_->manifest.runtimeArchitecture->layerCount ||
        impl_->manifest.experts.empty()) {
        throw std::invalid_argument("packed artifact is not a complete MoE model");
    }
    if (impl_->manifest.config.capabilities.sharedExperts) {
        throw std::invalid_argument(
            "Qwen2 shared-expert execution is not implemented; refusing partial logits");
    }
    impl_->profiler = std::make_shared<Profiler>();
    if (configuration.device.type == tensor::DeviceType::CUDA) {
        auto tensorBackend = std::make_shared<tensor::CudaTensorBackend>(
            configuration.device.ordinal, impl_->profiler);
        auto computeBackend = std::make_shared<backend::CudaBackend>(
            configuration.device.ordinal);
        if (!tensorBackend->available() || !computeBackend->isAvailable()) {
            throw std::runtime_error("requested CUDA runtime is unavailable");
        }
        impl_->tensors = std::move(tensorBackend);
        impl_->transferBackend = std::move(computeBackend);
    } else {
        impl_->tensors = std::make_shared<tensor::CpuTensorBackend>();
        impl_->transferBackend = std::make_shared<backend::CpuBackend>();
    }

    storage::MappedFile packedData(artifact / "experts.bin");
    RuntimeTensorMap runtimeTensors;
    tensor::CpuTensorBackend cpu;
    for (const auto& name : staticTensorNames(impl_->manifest)) {
        const auto* metadata = impl_->manifest.findTensor(name);
        if (!metadata || metadata->sourceFile != "experts.bin") {
            throw std::invalid_argument("runtime tensor is outside the packed artifact");
        }
        const auto bytes = packedData.view(metadata->offset, metadata->size);
        auto values = tensor::precision::DTypeConverter::toFp32(bytes, metadata->dtype);
        if (values.size() != metadata->shape.elementCount()) {
            throw std::invalid_argument("runtime tensor storage size does not match shape");
        }
        auto host = cpu.allocateTensor(metadata->shape, tensor::DType::FP32);
        std::memcpy(host.data(), values.data(), values.size() * sizeof(float));
        tensor::Tensor loaded;
        if (configuration.device.type == tensor::DeviceType::CPU) {
            loaded = std::move(host);
        } else {
            loaded = impl_->tensors->allocateTensor(metadata->shape,
                                                    tensor::DType::FP32);
            impl_->tensors->copyTensor(host.view(), loaded.view());
        }
        if (impl_->staticStorageBytes > std::numeric_limits<std::size_t>::max() -
                                            static_cast<std::size_t>(metadata->size) ||
            impl_->staticExecutionBytes > std::numeric_limits<std::size_t>::max() -
                                              loaded.storageBytes()) {
            throw std::overflow_error("runtime tensor memory accounting overflow");
        }
        impl_->staticStorageBytes += static_cast<std::size_t>(metadata->size);
        impl_->staticExecutionBytes += loaded.storageBytes();
        runtimeTensors.add(name, std::move(loaded));
    }

    impl_->store = std::make_shared<storage::ExpertStore>(artifact);
    impl_->loader = std::make_shared<storage::DiskLoader>(impl_->store);
    impl_->transfers = std::make_shared<TransferManager>(
        impl_->loader, impl_->transferBackend, configuration.transferWorkers);
    impl_->memory = std::make_unique<MemoryManager>(
        configuration.expertDeviceBudgetBytes, configuration.expertRamBudgetBytes);
    impl_->experts = std::make_unique<ExpertManager>(
        *impl_->memory, std::make_unique<LruCachePolicy>(), impl_->transfers);
    impl_->scheduler = std::make_shared<scheduler::Scheduler>(
        impl_->transfers, impl_->profiler, configuration.schedulerWorkers);
    for (const auto& record : impl_->store->index().records()) {
        if (record.size > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("expert size exceeds addressable runtime memory");
        }
        impl_->experts->registerExpert(
            {record.expert_id, record.layer_id, static_cast<std::size_t>(record.size),
             quantization(record.quantization_type), MemoryTier::Nvme});
        impl_->scheduler->registerExpert(record.layer_id, record.expert_id);
    }
    impl_->history = std::make_shared<prediction::ExpertHistory>();
    impl_->transitions = std::make_shared<prediction::TransitionDatabase>();
    impl_->predictor = std::make_shared<prediction::ExpertPredictor>(impl_->transitions);
    std::shared_ptr<router::RouterBackend> routerBackend;
    if (configuration.device.type == tensor::DeviceType::CUDA) {
        routerBackend = std::make_shared<router::CudaRouterBackend>(impl_->tensors);
    } else {
        routerBackend = std::make_shared<router::CpuRouterBackend>();
    }
    impl_->router = std::make_shared<router::Router>(
        impl_->manifest.router.config, std::move(routerBackend));
    impl_->expertExecutor = std::make_shared<ExpertMlpExecutor>(
        impl_->tensors, tensor::activation::ActivationType::SiLU, impl_->profiler);
    impl_->moeRuntime = std::make_shared<hypermoe::runtime::MoERuntime>(
        impl_->router, impl_->scheduler, *impl_->experts,
        makeExpertMappings(impl_->manifest), impl_->tensors,
        impl_->expertExecutor, impl_->history, impl_->predictor);
    impl_->moe = std::make_shared<transformer::MoELayer>(
        impl_->moeRuntime, impl_->tensors);
    if (configuration.device.type == tensor::DeviceType::CUDA) {
        impl_->attention = std::make_shared<transformer::attention::CudaAttention>(
            impl_->tensors);
    } else {
        impl_->attention = std::make_shared<transformer::attention::CpuAttention>(
            impl_->tensors);
    }
    const auto& architecture = *impl_->manifest.runtimeArchitecture;
    impl_->inputNorm = std::make_shared<transformer::norm::RMSNorm>(
        impl_->tensors, architecture.hiddenDimension,
        architecture.inputNormalization.epsilon);
    impl_->postAttentionNorm = std::make_shared<transformer::norm::RMSNorm>(
        impl_->tensors, architecture.hiddenDimension,
        architecture.postAttentionNormalization.epsilon);
    impl_->transformer = std::make_shared<TransformerModelRuntime>(
        impl_->manifest, std::move(runtimeTensors), impl_->attention,
        impl_->inputNorm, impl_->postAttentionNorm, impl_->moe, impl_->tensors);
    impl_->model = std::make_shared<ModelRuntime>(
        impl_->manifest, impl_->transformer, impl_->tensors);
    impl_->profiler->observeMemory(
        configuration.device.type == tensor::DeviceType::CUDA
            ? impl_->staticExecutionBytes : 0,
        configuration.device.type == tensor::DeviceType::CPU
            ? impl_->staticExecutionBytes : 0);
    if (configuration.device.type == tensor::DeviceType::CUDA) {
        impl_->profiler->observeGpuMemory(impl_->staticExecutionBytes);
    }
}

PackedModelRuntime::~PackedModelRuntime() = default;

ModelForwardResult PackedModelRuntime::forward(
    std::span<const std::uint32_t> tokenIds, std::uint64_t sequencePosition) {
    hypermoe::runtime::InferenceContext context;
    context.batchSize = tokenIds.size();
    context.sequencePosition = sequencePosition;
    context.hiddenDimension = architecture().hiddenDimension;
    auto result = impl_->model->forward(context, tokenIds);
    recordExecution(*impl_->profiler, result, impl_->memory->snapshot(),
                    device(), impl_->staticExecutionBytes);
    return result;
}

ModelForwardResult PackedModelRuntime::forward(
    std::span<const std::uint32_t> tokenIds, std::uint64_t sequencePosition,
    hypermoe::runtime::cache::KVCacheBase& cache) {
    hypermoe::runtime::InferenceContext context;
    context.batchSize = tokenIds.size();
    context.sequencePosition = sequencePosition;
    context.hiddenDimension = architecture().hiddenDimension;
    auto result = impl_->model->forward(context, tokenIds, cache);
    recordExecution(*impl_->profiler, result, impl_->memory->snapshot(),
                    device(), impl_->staticExecutionBytes);
    return result;
}

std::shared_ptr<hypermoe::runtime::cache::KVCacheBase>
PackedModelRuntime::createKVCache(std::size_t maximumSequenceLength) const {
    const auto& architecture = this->architecture();
    if (maximumSequenceLength == 0) {
        throw std::invalid_argument("KV cache sequence capacity must be nonzero");
    }
    if (device().type == tensor::DeviceType::CUDA) {
        return std::make_shared<hypermoe::runtime::cache::CudaKVCache>(
            impl_->tensors, architecture.layerCount, maximumSequenceLength,
            architecture.keyValueHeads, architecture.headDimension);
    }
    return std::make_shared<hypermoe::runtime::cache::KVCache>(
        architecture.layerCount, maximumSequenceLength,
        architecture.keyValueHeads, architecture.headDimension);
}

const models::ModelManifest& PackedModelRuntime::manifest() const noexcept {
    return impl_->manifest;
}

const ModelArchitecture& PackedModelRuntime::architecture() const noexcept {
    return impl_->model->architecture();
}

tensor::Device PackedModelRuntime::device() const noexcept {
    return impl_->tensors->device();
}

tensor::Tensor PackedModelRuntime::materializeHost(tensor::TensorView value) const {
    return impl_->model->materializeHost(value);
}

PackedRuntimeSnapshot PackedModelRuntime::snapshot() const {
    return {impl_->memory->snapshot(), impl_->experts->stats(),
            impl_->profiler->snapshot(), impl_->history->snapshot(),
            impl_->transferBackend->stats(), impl_->staticStorageBytes,
            impl_->staticExecutionBytes};
}

} // namespace hypermoe::models::runtime
