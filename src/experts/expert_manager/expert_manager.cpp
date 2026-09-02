#include "hypermoe/experts/expert_manager.hpp"

#include "memory/TransferManager.hpp"

#include <chrono>
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
    if (experts_.contains(expert.id)) {
        throw std::invalid_argument("duplicate expert id");
    }

    const auto initialTier = expert.location;
    expert.location = MemoryTier::Nvme;
    experts_.emplace(expert.id,
                     ManagedExpert{expert, std::nullopt, nullptr, nullptr});
    if (initialTier != MemoryTier::Nvme) {
        try {
            moveExpertLocked(expert.id, initialTier, {expert.id});
        } catch (...) {
            auto& inserted = experts_.at(expert.id);
            if (inserted.allocation) {
                (void)memory_.release(inserted.allocation->id);
            }
            experts_.erase(expert.id);
            throw;
        }
    }
}

ExpertRequestResult ExpertManager::requestExpert(ExpertId id) {
    std::scoped_lock lock(mutex_);
    auto& managed = requireExpertLocked(id);
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

    moveExpertLocked(id, MemoryTier::Vram, {id});
    if (transfers_ && sourceTier != MemoryTier::Vram) {
        result.simulatedLatencyMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
    }
    policy_->onAccess(id);
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
    result.expert = requireExpertLocked(id).metadata;
    return result;
}

void ExpertManager::moveExpert(ExpertId id, MemoryTier destination) {
    std::scoped_lock lock(mutex_);
    moveExpertLocked(id, destination, {id});
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
        if (tier == MemoryTier::Vram) {
            try {
                moveExpertLocked(*victim, MemoryTier::Ram, {*victim});
            } catch (const std::runtime_error&) {
                moveExpertLocked(*victim, MemoryTier::Nvme, {*victim});
            }
        } else {
            moveExpertLocked(*victim, MemoryTier::Nvme, {*victim});
        }
        ++evicted;
    }
    return evicted;
}

std::optional<Expert> ExpertManager::findExpert(ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(id);
    if (it == experts_.end()) {
        return std::nullopt;
    }
    return it->second.metadata;
}

std::shared_ptr<const std::vector<std::byte>>
ExpertManager::residentWeights(ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(id);
    if (it == experts_.end()) {
        throw std::out_of_range("unknown expert id");
    }
    return it->second.weights;
}

std::shared_ptr<backend::DeviceBuffer>
ExpertManager::residentDeviceWeights(ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto it = experts_.find(id);
    if (it == experts_.end()) {
        throw std::out_of_range("unknown expert id");
    }
    return it->second.deviceWeights;
}

std::size_t ExpertManager::expertCount() const {
    std::scoped_lock lock(mutex_);
    return experts_.size();
}

ExpertManagerStats ExpertManager::stats() const {
    std::scoped_lock lock(mutex_);
    return stats_;
}

void ExpertManager::moveExpertLocked(ExpertId id,
                                     MemoryTier destination,
                                     const std::unordered_set<ExpertId>& pinned) {
    auto& managed = requireExpertLocked(id);
    const auto source = managed.metadata.location;
    if (source == destination) {
        return;
    }

    // All cold loads stage through RAM; the DiskStore implementation will make
    // this an actual asynchronous range read in Phase 2.
    if (source == MemoryTier::Nvme && destination == MemoryTier::Vram) {
        moveExpertLocked(id, MemoryTier::Ram, pinned);
        moveExpertLocked(id, MemoryTier::Vram, pinned);
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
        policy_->onEvict(id, source);
        managed.allocation.reset();
        managed.weights.reset();
        managed.deviceWeights.reset();
        managed.metadata.location = MemoryTier::Nvme;
        return;
    }

    makeRoomLocked(destination, managed.metadata.sizeBytes, pinned);
    auto allocation = memory_.allocate(
        destination, managed.metadata.sizeBytes, "expert:" + std::to_string(id));
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
        policy_->onEvict(id, source);
    }
    managed.allocation = std::move(allocation);
    managed.weights = std::move(destinationWeights);
    managed.deviceWeights = std::move(destinationDeviceWeights);
    managed.metadata.location = destination;
    policy_->onResident(id, destination);
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
        if (tier == MemoryTier::Vram) {
            try {
                moveExpertLocked(*victim, MemoryTier::Ram, nextPinned);
            } catch (const std::runtime_error&) {
                moveExpertLocked(*victim, MemoryTier::Nvme, nextPinned);
            }
        } else {
            moveExpertLocked(*victim, MemoryTier::Nvme, nextPinned);
        }
    }
}

std::vector<ExpertId> ExpertManager::candidatesLocked(MemoryTier tier) const {
    std::vector<ExpertId> candidates;
    candidates.reserve(experts_.size());
    for (const auto& [id, expert] : experts_) {
        if (expert.metadata.location == tier) {
            candidates.push_back(id);
        }
    }
    return candidates;
}

ExpertManager::ManagedExpert& ExpertManager::requireExpertLocked(ExpertId id) {
    const auto it = experts_.find(id);
    if (it == experts_.end()) {
        throw std::out_of_range("unknown expert id");
    }
    return it->second;
}

} // namespace hypermoe
