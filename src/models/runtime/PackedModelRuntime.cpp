#include "models/runtime/PackedModelRuntime.hpp"

#include "backend/CpuBackend.hpp"
#include "backend/CudaBackend.hpp"
#include "cache/LRUPolicy.hpp"
#include "cache/HybridPolicy.hpp"
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
#include <charconv>
#include <cstring>
#include <cmath>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypermoe::models::runtime {
namespace {

std::size_t checkedMultiply(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error("expert device headroom accounting overflows");
    }
    return left * right;
}

std::size_t checkedAdd(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error("expert device headroom accounting overflows");
    }
    return left + right;
}

std::size_t growthSafeCacheBytes(const hypermoe::runtime::cache::KVCacheBase& cache) {
    // During geometric growth old and expanded buffers coexist. Twice the
    // maximum logical cache size conservatively covers that transient peak.
    return checkedMultiply(cache.maximumMemoryUsageBytes(), 2U);
}

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
            const auto* metadata = manifest.findTensor(projection.tensorName);
            if (!metadata) {
                throw std::invalid_argument("packed expert projection tensor is missing");
            }
            result.add(expert.layerId, expert.expertId, type,
                       {metadata->name, projection.shape, metadata->dtype,
                        metadata->quantization
                            ? std::optional{
                                  tensor::quantization::QuantizedDType::INT8}
                            : std::nullopt,
                        projection.offset, projection.size,
                        expert.layerId, expert.expertId,
                        metadata->quantization});
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
    ExpertDeviceBudgetPlan expertDeviceBudget;
    mutable std::mutex cacheReservationMutex;
    std::vector<std::weak_ptr<hypermoe::runtime::cache::KVCacheBase>> reservedCaches;
    std::size_t largestExpertBytes{};

    void validateWorkspace(std::size_t tokens, std::size_t keys) const {
        if (!configuration.automaticExpertDeviceBudget) return;
        const auto& architecture = *manifest.runtimeArchitecture;
        const auto projectionDimension = std::max(architecture.headDimension,
                                                 architecture.projectionHeadDimension);
        std::size_t intermediate = architecture.hiddenDimension;
        for (const auto& expert : manifest.experts) {
            for (const auto dimension : expert.gate.shape.dimensions()) {
                intermediate = std::max(intermediate, dimension);
            }
        }
        // Conservative single-forward envelope for retained layer tensors,
        // QKV/context, scores+softmax, router/gather/scatter, logits and INT8
        // conversion scratch. Wide projections and GQA use their own dimensions.
        auto perToken = checkedMultiply(architecture.hiddenDimension,
            checkedAdd(checkedMultiply(architecture.layerCount,
                checkedAdd(5U, checkedMultiply(architecture.topK, 2U))), 16U));
        perToken = checkedAdd(perToken, checkedMultiply(architecture.vocabularySize, 2U));
        perToken = checkedAdd(perToken, checkedMultiply(
            checkedMultiply(architecture.attentionHeads, projectionDimension), 8U));
        perToken = checkedAdd(perToken, checkedMultiply(intermediate, 8U));
        perToken = checkedAdd(perToken, checkedMultiply(architecture.expertCount, 4U));
        auto required = checkedMultiply(checkedMultiply(perToken, tokens), sizeof(float));
        required = checkedAdd(required, checkedMultiply(
            checkedMultiply(checkedMultiply(architecture.attentionHeads, tokens), keys),
            2U * sizeof(float)));
        required = checkedAdd(required, checkedMultiply(
            checkedMultiply(checkedMultiply(architecture.keyValueHeads, projectionDimension), keys),
            8U * sizeof(float)));
        required = checkedAdd(required, checkedMultiply(largestExpertBytes, 4U));
        if (required > configuration.expertDeviceReservations.workspaceBytes) {
            throw std::invalid_argument("forward dimensions exceed automatic expert budget workspace reservation");
        }
    }
};

void PackedRuntimeConfiguration::validate() const {
    if (expertDeviceBudgetBytes == 0 || expertRamBudgetBytes == 0 ||
        transferWorkers == 0 || schedulerWorkers == 0 || device.ordinal < 0 ||
        (device.type == tensor::DeviceType::CPU && device.ordinal != 0) ||
        !std::isfinite(minimumPrefetchConfidence) ||
        minimumPrefetchConfidence < 0.0 || minimumPrefetchConfidence > 1.0) {
        throw std::invalid_argument("packed runtime configuration is invalid");
    }
    if (automaticExpertDeviceBudget && device.type != tensor::DeviceType::CUDA) {
        throw std::invalid_argument("automatic expert device sizing requires CUDA; CPU residency remains RAM-only");
    }
}

std::size_t PackedRuntimeConfiguration::parseBudgetBytes(std::string_view text) {
    if (text.empty()) {
        throw std::invalid_argument("expert budget cannot be empty");
    }
    std::size_t value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end == text.data() || value == 0) {
        throw std::invalid_argument("expert budget must be a positive integer with an optional unit");
    }
    const std::string_view suffix{end, static_cast<std::size_t>(text.data() + text.size() - end)};
    std::size_t multiplier{1};
    if (suffix == "KiB") multiplier = 1024U;
    else if (suffix == "MiB") multiplier = 1024U * 1024U;
    else if (suffix == "GiB") multiplier = 1024U * 1024U * 1024U;
    else if (suffix == "KB") multiplier = 1000U;
    else if (suffix == "MB") multiplier = 1000U * 1000U;
    else if (suffix == "GB") multiplier = 1000U * 1000U * 1000U;
    else if (!suffix.empty() && suffix != "B") {
        throw std::invalid_argument("expert budget unit must be B, KB, MB, GB, KiB, MiB, or GiB");
    }
    if (value > std::numeric_limits<std::size_t>::max() / multiplier) {
        throw std::overflow_error("expert budget exceeds addressable memory");
    }
    return value * multiplier;
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
        impl_->tensors = std::make_shared<tensor::CpuTensorBackend>(impl_->profiler);
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
    std::size_t largestExpertBytes{};
    for (const auto& record : impl_->store->index().records()) {
        if (record.size > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("expert size exceeds addressable runtime memory");
        }
        largestExpertBytes = std::max(largestExpertBytes, static_cast<std::size_t>(record.size));
    }
    impl_->expertDeviceBudget.configuredBytes = configuration.expertDeviceBudgetBytes;
    impl_->largestExpertBytes = largestExpertBytes;
    impl_->expertDeviceBudget.effectiveBytes = configuration.expertDeviceBudgetBytes;
    if (configuration.device.type == tensor::DeviceType::CUDA) {
        // Cover worker-owned transfers plus per-expert pool alignment. The
        // existing pool can ALSO retain 512MiB of released blocks, accounted
        // separately by stagingPoolBytes. No additional transfer system.
        const auto alignedLargest = checkedAdd(largestExpertBytes, 255U) / 256U * 256U;
        const auto transferAllowance = checkedAdd(
            checkedMultiply(alignedLargest, configuration.transferWorkers),
            checkedMultiply(impl_->store->index().records().size(), 255U));
        impl_->expertDeviceBudget = planExpertDeviceBudget(
            configuration.expertDeviceBudgetBytes, configuration.automaticExpertDeviceBudget,
            impl_->transferBackend->getMemoryInfo(), impl_->staticExecutionBytes,
            configuration.expertDeviceReservations, largestExpertBytes, transferAllowance);
    }
    const auto executionBudget = configuration.device.type == tensor::DeviceType::CUDA
        ? impl_->expertDeviceBudget.effectiveBytes : configuration.expertRamBudgetBytes;
    for (const auto& record : impl_->store->index().records()) {
        if (record.size > executionBudget) {
            throw std::invalid_argument("expert execution budget cannot hold the largest indexed expert");
        }
    }
    impl_->loader = std::make_shared<storage::DiskLoader>(impl_->store);
    impl_->transfers = std::make_shared<TransferManager>(
        impl_->loader, impl_->transferBackend, configuration.transferWorkers);
    impl_->memory = std::make_unique<MemoryManager>(
        impl_->expertDeviceBudget.effectiveBytes, configuration.expertRamBudgetBytes);
    std::unique_ptr<CachePolicy> cachePolicy;
    if (configuration.adaptiveResidency) {
        cachePolicy = std::make_unique<HybridPolicy>(true);
    } else {
        cachePolicy = std::make_unique<LruCachePolicy>();
    }
    impl_->experts = std::make_unique<ExpertManager>(
        *impl_->memory, std::move(cachePolicy), impl_->transfers);
    impl_->scheduler = std::make_shared<scheduler::Scheduler>(
        impl_->transfers, impl_->profiler, configuration.schedulerWorkers,
        MemoryTier::Ram, configuration.expertRamBudgetBytes);
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
    if (configuration.adaptivePrediction) {
        prediction::AdaptivePredictionConfig predictorConfiguration;
        predictorConfiguration.minimumPrefetchConfidence =
            configuration.minimumPrefetchConfidence;
        impl_->predictor = std::make_shared<prediction::ExpertPredictor>(
            impl_->transitions, predictorConfiguration, impl_->profiler);
    }
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
        impl_->expertExecutor, impl_->history, impl_->predictor,
        configuration.transferComputeOverlap);
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
    impl_->validateWorkspace(tokenIds.size(), tokenIds.size());
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
    if (impl_->configuration.automaticExpertDeviceBudget) {
        std::scoped_lock lock(impl_->cacheReservationMutex);
        const auto registered = std::any_of(
            impl_->reservedCaches.begin(), impl_->reservedCaches.end(),
            [&cache](const auto& weak) { return weak.lock().get() == &cache; });
        if (!registered) {
            throw std::invalid_argument("automatic residency sizing requires a KV cache reserved by this packed runtime");
        }
    }
    std::size_t pastTokens{};
    if (impl_->configuration.automaticExpertDeviceBudget) {
        for (std::size_t layer = 0; layer < cache.layerCount(); ++layer) {
            pastTokens = std::max(pastTokens, cache.tokenCount(layer));
        }
    }
    impl_->validateWorkspace(tokenIds.size(), checkedAdd(pastTokens, tokenIds.size()));
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
        auto cache = std::make_shared<hypermoe::runtime::cache::CudaKVCache>(
            impl_->tensors, architecture.layerCount, maximumSequenceLength,
            architecture.keyValueHeads, architecture.headDimension);
        if (impl_->configuration.automaticExpertDeviceBudget) {
            std::scoped_lock lock(impl_->cacheReservationMutex);
            std::erase_if(impl_->reservedCaches, [](const auto& weak) { return weak.expired(); });
            auto requested = growthSafeCacheBytes(*cache);
            for (const auto& weak : impl_->reservedCaches) {
                if (const auto existing = weak.lock()) {
                    requested = checkedAdd(requested, growthSafeCacheBytes(*existing));
                }
            }
            if (requested > impl_->configuration.expertDeviceReservations.kvCacheBytes) {
                throw std::invalid_argument("KV cache capacities exceed automatic expert budget KV reservation (including growth headroom)");
            }
            impl_->reservedCaches.push_back(cache);
        }
        return cache;
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
    const auto* cuda = dynamic_cast<const tensor::CudaTensorBackend*>(impl_->tensors.get());
    return {impl_->memory->snapshot(), impl_->experts->stats(),
            impl_->profiler->snapshot(), impl_->history->snapshot(),
            impl_->predictor ? impl_->predictor->qualitySnapshot()
                             : prediction::PredictionQualitySnapshot{},
            impl_->experts->residencySnapshot(),
            impl_->transferBackend->stats(), impl_->staticStorageBytes,
            impl_->staticExecutionBytes,
            cuda ? cuda->backendStats() : backend::BackendStats{},
            impl_->configuration.transferComputeOverlap,
            impl_->expertDeviceBudget, impl_->transferBackend->getMemoryInfo()};
}

hypermoe::runtime::metrics::RuntimeMetricsSnapshot
PackedModelRuntime::metrics() const {
    const auto current = snapshot();
    return hypermoe::runtime::metrics::collectRuntimeMetrics(
        current.expertMemory, current.history, current.prediction,
        current.profiler, current.residency);
}

} // namespace hypermoe::models::runtime
