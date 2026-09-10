#include "runtime/cache/KVCacheManager.hpp"

#include "runtime/cache/CudaKVCache.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::runtime::cache {

KVCacheManager::KVCacheManager(std::size_t layerCount,
                               std::size_t maximumSequenceLength,
                               std::size_t keyValueHeads,
                               std::size_t headDimension,
                               std::size_t memoryLimitBytes,
                               std::shared_ptr<tensor::TensorBackend> backend)
    : layerCount_(layerCount),
      maximumSequenceLength_(maximumSequenceLength),
      keyValueHeads_(keyValueHeads),
      headDimension_(headDimension),
      memoryLimitBytes_(memoryLimitBytes), backend_(std::move(backend)) {
    if (backend_ && (!backend_->available() ||
                     backend_->device().type != tensor::DeviceType::CUDA)) {
        throw std::invalid_argument(
            "KV cache manager device backend must be available CUDA");
    }
    KVCache prototype(layerCount_, maximumSequenceLength_, keyValueHeads_,
                      headDimension_);
    bytesPerSession_ = prototype.maximumMemoryUsageBytes();
    if (memoryLimitBytes_ == 0 || bytesPerSession_ > memoryLimitBytes_) {
        throw std::invalid_argument(
            "KV cache memory limit cannot hold one maximum-length session");
    }
}

KVCacheAllocation KVCacheManager::allocateSession() {
    std::scoped_lock lock(mutex_);
    if (reservedBytes_ > memoryLimitBytes_ - bytesPerSession_) {
        throw std::runtime_error("KV cache memory limit is exhausted");
    }
    if (nextSessionId_ == 0) {
        throw std::overflow_error("KV cache session ID space is exhausted");
    }
    const auto sessionId = nextSessionId_++;
    std::shared_ptr<KVCacheBase> cache;
    if (backend_) {
        cache = std::make_shared<CudaKVCache>(
            backend_, layerCount_, maximumSequenceLength_, keyValueHeads_, headDimension_);
    } else {
        cache = std::make_shared<KVCache>(
            layerCount_, maximumSequenceLength_, keyValueHeads_, headDimension_);
    }
    if (!sessions_.emplace(sessionId, cache).second) {
        throw std::logic_error("KV cache session ID collision");
    }
    reservedBytes_ += bytesPerSession_;
    return {sessionId, std::move(cache), bytesPerSession_};
}

void KVCacheManager::releaseSession(KVCacheSessionId sessionId) {
    std::scoped_lock lock(mutex_);
    if (sessions_.erase(sessionId) == 0) {
        throw std::out_of_range("KV cache session is not active");
    }
    reservedBytes_ -= bytesPerSession_;
}

KVCacheManagerStats KVCacheManager::stats() const {
    std::scoped_lock lock(mutex_);
    std::size_t committed{};
    for (const auto& [sessionId, cache] : sessions_) {
        (void)sessionId;
        const auto bytes = cache->memoryUsageBytes();
        if (committed > std::numeric_limits<std::size_t>::max() - bytes) {
            throw std::overflow_error("KV cache committed-byte accounting overflows");
        }
        committed += bytes;
    }
    if (committed > peakCommittedBytes_) peakCommittedBytes_ = committed;
    return {sessions_.size(), committed, reservedBytes_, peakCommittedBytes_,
            memoryLimitBytes_};
}

std::size_t KVCacheManager::layerCount() const noexcept { return layerCount_; }
std::size_t KVCacheManager::maximumSequenceLength() const noexcept {
    return maximumSequenceLength_;
}
std::size_t KVCacheManager::keyValueHeads() const noexcept {
    return keyValueHeads_;
}
std::size_t KVCacheManager::headDimension() const noexcept {
    return headDimension_;
}
std::size_t KVCacheManager::bytesPerSession() const noexcept {
    return bytesPerSession_;
}
tensor::Device KVCacheManager::device() const noexcept {
    return backend_ ? backend_->device() : tensor::Device::cpu();
}

} // namespace hypermoe::runtime::cache
