#pragma once

#include "runtime/cache/KVCache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace hypermoe::runtime::cache {

using KVCacheSessionId = std::uint64_t;

struct KVCacheAllocation {
    KVCacheSessionId sessionId{};
    std::shared_ptr<KVCache> cache;
    std::size_t reservedBytes{};
};

struct KVCacheManagerStats {
    std::size_t activeSessions{};
    std::size_t committedBytes{};
    std::size_t reservedBytes{};
    std::size_t peakCommittedBytes{};
    std::size_t memoryLimitBytes{};
};

class KVCacheManager {
public:
    KVCacheManager(std::size_t layerCount,
                   std::size_t maximumSequenceLength,
                   std::size_t keyValueHeads,
                   std::size_t headDimension,
                   std::size_t memoryLimitBytes);

    [[nodiscard]] KVCacheAllocation allocateSession();
    void releaseSession(KVCacheSessionId sessionId);
    [[nodiscard]] KVCacheManagerStats stats() const;

    [[nodiscard]] std::size_t layerCount() const noexcept;
    [[nodiscard]] std::size_t maximumSequenceLength() const noexcept;
    [[nodiscard]] std::size_t keyValueHeads() const noexcept;
    [[nodiscard]] std::size_t headDimension() const noexcept;
    [[nodiscard]] std::size_t bytesPerSession() const noexcept;

private:
    std::size_t layerCount_{};
    std::size_t maximumSequenceLength_{};
    std::size_t keyValueHeads_{};
    std::size_t headDimension_{};
    std::size_t memoryLimitBytes_{};
    std::size_t bytesPerSession_{};
    mutable std::size_t peakCommittedBytes_{};
    std::size_t reservedBytes_{};
    KVCacheSessionId nextSessionId_{1};
    mutable std::mutex mutex_;
    std::unordered_map<KVCacheSessionId, std::shared_ptr<KVCache>> sessions_;
};

} // namespace hypermoe::runtime::cache
