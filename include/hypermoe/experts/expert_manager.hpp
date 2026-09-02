#pragma once

#include "hypermoe/experts/cache_policy.hpp"
#include "hypermoe/memory/memory_manager.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hypermoe {

class TransferManager;
namespace backend {
class DeviceBuffer;
}

enum class RequestSource {
    VramHit,
    RamHit,
    NvmeLoad,
};

struct ExpertRequestResult {
    Expert expert;
    RequestSource source{RequestSource::NvmeLoad};
    double simulatedLatencyMs{};
};

struct ExpertManagerStats {
    std::uint64_t requests{};
    std::uint64_t vramHits{};
    std::uint64_t ramHits{};
    std::uint64_t nvmeLoads{};
    std::uint64_t vramEvictions{};
    std::uint64_t ramEvictions{};
    std::uint64_t nvmeBytesRead{};
    double simulatedLoadingLatencyMs{};

    [[nodiscard]] double vramHitRate() const noexcept;
};

class ExpertManager {
public:
    ExpertManager(MemoryManager& memory, std::unique_ptr<CachePolicy> policy);
    ExpertManager(MemoryManager& memory,
                  std::unique_ptr<CachePolicy> policy,
                  std::shared_ptr<TransferManager> transfers);

    void registerExpert(Expert expert);
    [[nodiscard]] ExpertRequestResult requestExpert(ExpertId id);
    void moveExpert(ExpertId id, MemoryTier destination);
    [[nodiscard]] std::size_t evictUntilWithin(MemoryTier tier,
                                               std::size_t maximumUsedBytes);

    [[nodiscard]] std::optional<Expert> findExpert(ExpertId id) const;
    [[nodiscard]] std::shared_ptr<const std::vector<std::byte>>
    residentWeights(ExpertId id) const;
    [[nodiscard]] std::shared_ptr<backend::DeviceBuffer>
    residentDeviceWeights(ExpertId id) const;
    [[nodiscard]] std::size_t expertCount() const;
    [[nodiscard]] ExpertManagerStats stats() const;

private:
    struct ManagedExpert {
        Expert metadata;
        std::optional<MemoryAllocation> allocation;
        std::shared_ptr<const std::vector<std::byte>> weights;
        std::shared_ptr<backend::DeviceBuffer> deviceWeights;
    };

    void moveExpertLocked(ExpertId id,
                          MemoryTier destination,
                          const std::unordered_set<ExpertId>& pinned);
    void makeRoomLocked(MemoryTier tier,
                        std::size_t requiredBytes,
                        const std::unordered_set<ExpertId>& pinned);
    [[nodiscard]] std::vector<ExpertId> candidatesLocked(MemoryTier tier) const;
    [[nodiscard]] ManagedExpert& requireExpertLocked(ExpertId id);

    MemoryManager& memory_;
    std::unique_ptr<CachePolicy> policy_;
    std::shared_ptr<TransferManager> transfers_;
    mutable std::mutex mutex_;
    std::unordered_map<ExpertId, ManagedExpert> experts_;
    ExpertManagerStats stats_;
};

} // namespace hypermoe
