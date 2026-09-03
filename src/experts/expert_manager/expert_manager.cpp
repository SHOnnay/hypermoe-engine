#include "hypermoe/experts/expert_manager.hpp"

#include "memory/TransferManager.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"
#include "tensor/quantization/QuantizedTensor.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypermoe {
namespace {

constexpr double kNvmeBaseLatencyMs = 0.18;
constexpr double kNvmeBytesPerMs = 2.5 * 1024.0 * 1024.0; // synthetic 2.5 GB/s
constexpr double kPcieBaseLatencyMs = 0.03;
constexpr double kPcieBytesPerMs = 12.0 * 1024.0 * 1024.0; // synthetic 12 GB/s

double nvmeLatency(std::size_t bytes) {
    return kNvmeBaseLatencyMs + static_cast<double>(bytes) / kNvmeBytesPerMs;
}

double pcieLatency(std::size_t bytes) {
    return kPcieBaseLatencyMs + static_cast<double>(bytes) / kPcieBytesPerMs;
}

} // namespace

double ExpertManagerStats::vramHitRate() const noexcept {
    return requests == 0 ? 0.0 : static_cast<double>(vramHits) / static_cast<double>(requests);
}

ExpertManager::ExpertManager(MemoryManager& memory, std::unique_ptr<CachePolicy> policy)
    : memory_(memory), policy_(std::move(policy)) {
    if (!policy_) {
        throw std::invalid_argument("ExpertManager requires a cache policy");
    }
}

ExpertManager::ExpertManager(MemoryManager& memory,
                             std::unique_ptr<CachePolicy> policy,
                             std::shared_ptr<TransferManager> transfers)
    : memory_(memory), policy_(std::move(policy)), transfers_(std::move(transfers)) {
    if (!policy_) {
        throw std::invalid_argument("ExpertManager requires a cache policy");
    }
    if (!transfers_) {
        throw std::invalid_argument("ExpertManager requires a TransferManager");
    }
}

void ExpertManager::registerExpert(Expert expert) {
    if (expert.sizeBytes == 0) {
        throw std::invalid_argument("expert size must be greater than zero");
    }

    std::scoped_lock lock(mutex_);
    const auto expertKey = key(expert.layer, expert.id);
    if (experts_.contains(expertKey)) {
        throw std::invalid_argument("duplicate layer/expert id");
    }
    if (nextPolicyId_ > std::numeric_limits<ExpertId>::max()) {
        throw std::overflow_error("cache policy expert ID space exhausted");
    }

    const auto initialTier = expert.location;
    const auto policyId = static_cast<ExpertId>(nextPolicyId_++);
    expert.location = MemoryTier::Nvme;
    experts_.emplace(expertKey,
                     ManagedExpert{expert, policyId, std::nullopt, nullptr, nullptr});
    policyExperts_.emplace(policyId, expertKey);
    legacyExperts_[expert.id].push_back(expertKey);
    if (initialTier != MemoryTier::Nvme) {
        try {
            moveExpertLocked(expert.layer, expert.id, initialTier, {policyId});
        } catch (...) {
            auto& inserted = experts_.at(expertKey);
            if (inserted.allocation) {
                (void)memory_.release(inserted.allocation->id);
            }
            experts_.erase(expertKey);
            policyExperts_.erase(policyId);
            auto& legacy = legacyExperts_.at(expert.id);
            legacy.erase(std::remove(legacy.begin(), legacy.end(), expertKey),
                         legacy.end());
            if (legacy.empty()) legacyExperts_.erase(expert.id);
            throw;
        }
    }
}

ExpertRequestResult ExpertManager::requestExpert(ExpertId id) {
    return requestExpert(resolveLegacyLayer(id), id);
}

ExpertRequestResult ExpertManager::requestExpert(LayerId layerId, ExpertId id) {
    std::scoped_lock lock(mutex_);
    auto& managed = requireExpertLocked(layerId, id);
    const auto sourceTier = managed.metadata.location;
    const auto size = managed.metadata.sizeBytes;
    const auto start = std::chrono::steady_clock::now();
    ExpertRequestResult result;
    if (sourceTier == MemoryTier::Vram) {
        result.source = RequestSource::VramHit;
    } else if (sourceTier == MemoryTier::Ram) {
        result.source = RequestSource::RamHit;
        result.simulatedLatencyMs = pcieLatency(size);
    } else {
        result.source = RequestSource::NvmeLoad;
        result.simulatedLatencyMs = nvmeLatency(size) + pcieLatency(size);
    }

    moveExpertLocked(layerId, id, MemoryTier::Vram, {managed.policyId});
    if (transfers_ && sourceTier != MemoryTier::Vram) {
        result.simulatedLatencyMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
    }
    policy_->onAccess(managed.policyId);
    ++stats_.requests;
    if (result.source == RequestSource::VramHit) {
        ++stats_.vramHits;
    } else if (result.source == RequestSource::RamHit) {
        ++stats_.ramHits;
    } else {
        ++stats_.nvmeLoads;
        stats_.nvmeBytesRead += size;
    }
    stats_.simulatedLoadingLatencyMs += result.simulatedLatencyMs;
    result.expert = requireExpertLocked(layerId, id).metadata;
    return result;
}

void ExpertManager::moveExpert(ExpertId id, MemoryTier destination) {
    moveExpert(resolveLegacyLayer(id), id, destination);
}

void ExpertManager::moveExpert(LayerId layerId,
                               ExpertId id,
                               MemoryTier destination) {
    std::scoped_lock lock(mutex_);
    auto& managed = requireExpertLocked(layerId, id);
    moveExpertLocked(layerId, id, destination, {managed.policyId});
}

std::size_t ExpertManager::evictUntilWithin(MemoryTier tier,
                                            std::size_t maximumUsedBytes) {
    if (tier == MemoryTier::Nvme) {
        throw std::invalid_argument("NVMe does not use MemoryManager capacity");
    }
    std::scoped_lock lock(mutex_);
    std::size_t evicted = 0;
    while (memory_.usage(tier).usedBytes > maximumUsedBytes) {
        const auto candidates = candidatesLocked(tier);
        const std::unordered_set<ExpertId> nonePinned;
        const auto victim = policy_->selectVictim(tier, candidates, nonePinned);
        if (!victim) {
            break;
        }
        const auto identity = policyExperts_.find(*victim);
        if (identity == policyExperts_.end()) {
            throw std::logic_error("cache policy selected an unknown expert");
        }
        const auto& victimExpert = experts_.at(identity->second).metadata;
        if (tier == MemoryTier::Vram) {
            try {
                moveExpertLocked(victimExpert.layer, victimExpert.id,
                                 MemoryTier::Ram, {*victim});
            } catch (const std::runtime_error&) {
                moveExpertLocked(victimExpert.layer, victimExpert.id,
                                 MemoryTier::Nvme, {*victim});
            }
        } else {
            moveExpertLocked(victimExpert.layer, victimExpert.id,
                             MemoryTier::Nvme, {*victim});
        }
        ++evicted;
    }
    return evicted;
}

std::optional<Expert> ExpertManager::findExpert(ExpertId id) const {
    try {
        return findExpert(resolveLegacyLayer(id), id);
    } catch (const std::out_of_range&) {
        return std::nullopt;
    }
}

std::optional<Expert> ExpertManager::findExpert(LayerId layerId, ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(key(layerId, id));
    if (it == experts_.end()) {
        return std::nullopt;
    }
    return it->second.metadata;
}

std::shared_ptr<const std::vector<std::byte>>
ExpertManager::residentWeights(ExpertId id) const {
    return residentWeights(resolveLegacyLayer(id), id);
}

std::shared_ptr<const std::vector<std::byte>>
ExpertManager::residentWeights(LayerId layerId, ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(key(layerId, id));
    if (it == experts_.end()) {
        throw std::out_of_range("unknown expert id");
    }
    return it->second.weights;
}

std::shared_ptr<backend::DeviceBuffer>
ExpertManager::residentDeviceWeights(ExpertId id) const {
    return residentDeviceWeights(resolveLegacyLayer(id), id);
}

std::shared_ptr<backend::DeviceBuffer>
ExpertManager::residentDeviceWeights(LayerId layerId, ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(key(layerId, id));
    if (it == experts_.end()) {
        throw std::out_of_range("unknown expert id");
    }
    return it->second.deviceWeights;
}

tensor::Tensor ExpertManager::residentDeviceTensor(
    ExpertId id, const tensor::Shape& shape, tensor::DType dtype) const {
    return residentDeviceTensor(resolveLegacyLayer(id), id, shape, dtype);
}

tensor::Tensor ExpertManager::residentDeviceTensor(
    LayerId layerId,
    ExpertId id,
    const tensor::Shape& shape,
    tensor::DType dtype) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(key(layerId, id));
    if (it == experts_.end()) throw std::out_of_range("unknown expert id");
    const auto& managed = it->second;
    if (managed.metadata.location != MemoryTier::Vram || !managed.deviceWeights) {
        throw std::logic_error("expert does not own a resident device buffer");
    }
    const auto elementBytes = tensor::sizeOf(dtype);
    if (elementBytes == 0 ||
        shape.storageElementCount() > managed.metadata.sizeBytes / elementBytes ||
        shape.storageElementCount() * elementBytes != managed.metadata.sizeBytes) {
        throw std::invalid_argument("tensor metadata does not match expert weight bytes");
    }
    const auto& buffer = managed.deviceWeights;
    const auto device = buffer->backend()->kind() == backend::BackendKind::Cuda
                            ? tensor::Device::cuda(buffer->backend()->deviceOrdinal())
                            : tensor::Device::cpu();
    return tensor::Tensor::fromDeviceBuffer(shape, dtype, device, buffer);
}

tensor::TensorView ExpertManager::residentDeviceTensorView(
    ExpertId id, const tensor::Shape& shape, tensor::DType dtype) const {
    return residentDeviceTensorView(resolveLegacyLayer(id), id, shape, dtype);
}

tensor::TensorView ExpertManager::residentDeviceTensorView(
    LayerId layerId,
    ExpertId id,
    const tensor::Shape& shape,
    tensor::DType dtype) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(key(layerId, id));
    if (it == experts_.end()) throw std::out_of_range("unknown expert id");
    const auto& managed = it->second;
    if (managed.metadata.location != MemoryTier::Vram || !managed.deviceWeights) {
        throw std::logic_error("expert does not own a resident device buffer");
    }
    const auto elementBytes = tensor::sizeOf(dtype);
    if (elementBytes == 0 ||
        shape.storageElementCount() > managed.metadata.sizeBytes / elementBytes ||
        shape.storageElementCount() * elementBytes != managed.metadata.sizeBytes) {
        throw std::invalid_argument("tensor view metadata does not match expert weight bytes");
    }
    const auto& buffer = managed.deviceWeights;
    const auto device = buffer->backend()->kind() == backend::BackendKind::Cuda
                            ? tensor::Device::cuda(buffer->backend()->deviceOrdinal())
                            : tensor::Device::cpu();
    return tensor::TensorView::fromDeviceBuffer(shape, dtype, device, buffer, false);
}

tensor::quantization::QuantizedTensor ExpertManager::residentQuantizedTensor(
    ExpertId id,
    const tensor::Shape& shape,
    tensor::quantization::QuantizedDType dtype,
    tensor::quantization::QuantizationParameters parameters) const {
    return residentQuantizedTensor(resolveLegacyLayer(id), id, shape, dtype,
                                   parameters);
}

tensor::quantization::QuantizedTensor ExpertManager::residentQuantizedTensor(
    LayerId layerId,
    ExpertId id,
    const tensor::Shape& shape,
    tensor::quantization::QuantizedDType dtype,
    tensor::quantization::QuantizationParameters parameters) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(key(layerId, id));
    if (it == experts_.end()) throw std::out_of_range("unknown expert id");
    const auto& managed = it->second;
    if (managed.metadata.location != MemoryTier::Vram || !managed.deviceWeights) {
        throw std::logic_error("expert does not own a resident device buffer");
    }
    const bool quantizationMatches =
        (dtype == tensor::quantization::QuantizedDType::INT8 &&
         managed.metadata.quantization == QuantizationType::Int8) ||
        (dtype == tensor::quantization::QuantizedDType::Q4 &&
         managed.metadata.quantization == QuantizationType::Q4);
    if (!quantizationMatches) {
        throw std::invalid_argument("quantized tensor dtype does not match expert metadata");
    }
    if (tensor::quantization::storageSizeBytes(shape, dtype) !=
        managed.metadata.sizeBytes) {
        throw std::invalid_argument(
            "quantized tensor metadata does not match expert weight bytes");
    }
    const auto& buffer = managed.deviceWeights;
    const auto device = buffer->backend()->kind() == backend::BackendKind::Cuda
                            ? tensor::Device::cuda(buffer->backend()->deviceOrdinal())
                            : tensor::Device::cpu();
    return tensor::quantization::QuantizedTensor::fromDeviceBuffer(
        shape, dtype, parameters, device, buffer);
}

void ExpertManager::adoptDeviceWeights(
    LayerId layerId,
    ExpertId id,
    std::shared_ptr<backend::DeviceBuffer> buffer) {
    if (!buffer || !*buffer || !buffer->backend() ||
        !buffer->backend()->isAvailable()) {
        throw std::invalid_argument("adopted expert buffer is unavailable");
    }
    std::scoped_lock lock(mutex_);
    auto& managed = requireExpertLocked(layerId, id);
    if (buffer->size() != managed.metadata.sizeBytes) {
        throw std::invalid_argument("adopted expert buffer size does not match metadata");
    }
    if (managed.metadata.location == MemoryTier::Vram) {
        if (managed.deviceWeights != buffer) {
            throw std::logic_error("expert already owns a different device buffer");
        }
        policy_->onAccess(managed.policyId);
        return;
    }

    makeRoomLocked(MemoryTier::Vram, managed.metadata.sizeBytes,
                   {managed.policyId});
    auto allocation = memory_.allocate(
        MemoryTier::Vram, managed.metadata.sizeBytes,
        "expert:" + std::to_string(layerId) + ":" + std::to_string(id));
    if (!allocation) throw std::runtime_error("failed reserving adopted expert memory");
    if (managed.allocation && !memory_.release(managed.allocation->id)) {
        (void)memory_.release(allocation->id);
        throw std::logic_error("expert owns an unknown source allocation");
    }
    if (managed.metadata.location != MemoryTier::Nvme) {
        policy_->onEvict(managed.policyId, managed.metadata.location);
    }
    managed.allocation = std::move(allocation);
    managed.weights.reset();
    managed.deviceWeights = std::move(buffer);
    managed.metadata.location = MemoryTier::Vram;
    policy_->onResident(managed.policyId, MemoryTier::Vram);
    policy_->onAccess(managed.policyId);
}

std::size_t ExpertManager::expertCount() const {
    std::scoped_lock lock(mutex_);
    return experts_.size();
}

ExpertManagerStats ExpertManager::stats() const {
    std::scoped_lock lock(mutex_);
    return stats_;
}

void ExpertManager::moveExpertLocked(LayerId layerId,
                                     ExpertId id,
                                     MemoryTier destination,
                                     const std::unordered_set<ExpertId>& pinned) {
    auto& managed = requireExpertLocked(layerId, id);
    const auto source = managed.metadata.location;
    if (source == destination) {
        return;
    }

    // All cold loads stage through RAM; the DiskStore implementation will make
    // this an actual asynchronous range read in Phase 2.
    if (source == MemoryTier::Nvme && destination == MemoryTier::Vram) {
        moveExpertLocked(layerId, id, MemoryTier::Ram, pinned);
        moveExpertLocked(layerId, id, MemoryTier::Vram, pinned);
        return;
    }

    if (destination == MemoryTier::Nvme) {
        if (managed.allocation && !memory_.release(managed.allocation->id)) {
            throw std::logic_error("expert owns an unknown memory allocation");
        }
        if (source == MemoryTier::Vram) {
            ++stats_.vramEvictions;
        } else if (source == MemoryTier::Ram) {
            ++stats_.ramEvictions;
        }
        policy_->onEvict(managed.policyId, source);
        managed.allocation.reset();
        managed.weights.reset();
        managed.deviceWeights.reset();
        managed.metadata.location = MemoryTier::Nvme;
        return;
    }

    makeRoomLocked(destination, managed.metadata.sizeBytes, pinned);
    auto allocation = memory_.allocate(
        destination, managed.metadata.sizeBytes,
        "expert:" + std::to_string(layerId) + ":" + std::to_string(id));
    if (!allocation) {
        throw std::runtime_error("memory reservation failed after eviction");
    }

    auto destinationWeights = managed.weights;
    auto destinationDeviceWeights = managed.deviceWeights;
    try {
        if (source == MemoryTier::Nvme && transfers_) {
            TransferRequest request;
            request.layerId = managed.metadata.layer;
            request.expertId = id;
            request.destination = MemoryTier::Ram;
            auto handle = transfers_->submit(std::move(request));
            auto loaded = handle.future().get();
            if (loaded.status != TransferStatus::Completed || !loaded.buffer) {
                throw std::runtime_error("expert disk transfer was cancelled");
            }
            if (loaded.buffer->size() != managed.metadata.sizeBytes) {
                throw std::runtime_error("expert index size does not match registered metadata");
            }
            destinationWeights = std::move(loaded.buffer);
            destinationDeviceWeights.reset();
        } else if (source == MemoryTier::Ram &&
                   destination == MemoryTier::Vram && transfers_) {
            TransferRequest request;
            request.layerId = managed.metadata.layer;
            request.expertId = id;
            request.source = MemoryTier::Ram;
            request.destination = MemoryTier::Vram;
            request.hostBuffer = managed.weights;
            auto transferred = transfers_->submit(std::move(request)).future().get();
            if (transferred.status != TransferStatus::Completed ||
                !transferred.deviceBuffer) {
                throw std::runtime_error("expert device transfer was cancelled");
            }
            destinationWeights.reset();
            destinationDeviceWeights = std::move(transferred.deviceBuffer);
        } else if (source == MemoryTier::Vram &&
                   destination == MemoryTier::Ram && transfers_) {
            TransferRequest request;
            request.layerId = managed.metadata.layer;
            request.expertId = id;
            request.source = MemoryTier::Vram;
            request.destination = MemoryTier::Ram;
            request.sourceDeviceBuffer = managed.deviceWeights;
            auto transferred = transfers_->submit(std::move(request)).future().get();
            if (transferred.status != TransferStatus::Completed || !transferred.buffer) {
                throw std::runtime_error("expert device download was cancelled");
            }
            destinationWeights = std::move(transferred.buffer);
            destinationDeviceWeights.reset();
        } else if (source != MemoryTier::Nvme && managed.weights) {
            destinationWeights =
                std::make_shared<const std::vector<std::byte>>(*managed.weights);
        }
    } catch (...) {
        (void)memory_.release(allocation->id);
        throw;
    }

    if (managed.allocation && !memory_.release(managed.allocation->id)) {
        (void)memory_.release(allocation->id);
        throw std::logic_error("expert owns an unknown source allocation");
    }

    if (source == MemoryTier::Vram) {
        ++stats_.vramEvictions;
    }
    if (source != MemoryTier::Nvme) {
        policy_->onEvict(managed.policyId, source);
    }
    managed.allocation = std::move(allocation);
    managed.weights = std::move(destinationWeights);
    managed.deviceWeights = std::move(destinationDeviceWeights);
    managed.metadata.location = destination;
    policy_->onResident(managed.policyId, destination);
}

void ExpertManager::makeRoomLocked(MemoryTier tier,
                                   std::size_t requiredBytes,
                                   const std::unordered_set<ExpertId>& pinned) {
    if (tier == MemoryTier::Nvme) {
        return;
    }
    if (requiredBytes > memory_.usage(tier).limitBytes) {
        throw std::runtime_error("expert is larger than destination tier capacity");
    }

    while (memory_.usage(tier).availableBytes() < requiredBytes) {
        const auto candidates = candidatesLocked(tier);
        const auto victim = policy_->selectVictim(tier, candidates, pinned);
        if (!victim) {
            throw std::runtime_error("no evictable expert can satisfy memory request");
        }

        auto nextPinned = pinned;
        nextPinned.insert(*victim);
        const auto identity = policyExperts_.find(*victim);
        if (identity == policyExperts_.end()) {
            throw std::logic_error("cache policy selected an unknown expert");
        }
        const auto& victimExpert = experts_.at(identity->second).metadata;
        if (tier == MemoryTier::Vram) {
            try {
                moveExpertLocked(victimExpert.layer, victimExpert.id,
                                 MemoryTier::Ram, nextPinned);
            } catch (const std::runtime_error&) {
                moveExpertLocked(victimExpert.layer, victimExpert.id,
                                 MemoryTier::Nvme, nextPinned);
            }
        } else {
            moveExpertLocked(victimExpert.layer, victimExpert.id,
                             MemoryTier::Nvme, nextPinned);
        }
    }
}

std::vector<ExpertId> ExpertManager::candidatesLocked(MemoryTier tier) const {
    std::vector<ExpertId> candidates;
    candidates.reserve(experts_.size());
    for (const auto& [expertKey, expert] : experts_) {
        (void)expertKey;
        if (expert.metadata.location == tier) {
            candidates.push_back(expert.policyId);
        }
    }
    return candidates;
}

ExpertManager::ManagedExpert& ExpertManager::requireExpertLocked(LayerId layerId,
                                                                 ExpertId id) {
    const auto it = experts_.find(key(layerId, id));
    if (it == experts_.end()) {
        throw std::out_of_range("unknown expert id");
    }
    return it->second;
}

LayerId ExpertManager::resolveLegacyLayer(ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto found = legacyExperts_.find(id);
    if (found == legacyExperts_.end() || found->second.empty()) {
        throw std::out_of_range("unknown expert id");
    }
    if (found->second.size() != 1) {
        throw std::logic_error(
            "expert id is ambiguous across layers; use the layer-aware API");
    }
    return experts_.at(found->second.front()).metadata.layer;
}

} // namespace hypermoe
