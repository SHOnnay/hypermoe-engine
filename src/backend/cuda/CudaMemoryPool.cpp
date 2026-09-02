#include "backend/cuda/CudaMemoryPool.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hypermoe::backend {
namespace {

constexpr std::size_t kAlignment = 256;

std::size_t alignedSize(std::size_t size) {
    if (size == 0) throw std::invalid_argument("memory-pool allocation must be nonzero");
    if (size > std::numeric_limits<std::size_t>::max() - (kAlignment - 1)) {
        throw std::overflow_error("memory-pool allocation size overflow");
    }
    return (size + kAlignment - 1) / kAlignment * kAlignment;
}

} // namespace

struct CudaBuffer::State {
    explicit State(std::shared_ptr<ComputeBackend> selectedBackend,
                   std::size_t cacheLimit)
        : backend(std::move(selectedBackend)), maximumCachedBytes(cacheLimit) {}

    std::shared_ptr<ComputeBackend> backend;
    std::size_t maximumCachedBytes{};
    mutable std::mutex mutex;
    std::multimap<std::size_t, void*> freeBlocks;
    std::unordered_map<void*, std::size_t> activeBlocks;
    CudaMemoryPoolStats statistics;
    bool accepting{true};
};

namespace {

struct Allocation {
    void* pointer{};
    std::size_t capacity{};
};

Allocation acquire(const std::shared_ptr<CudaBuffer::State>& state,
                   std::size_t requested) {
    const auto capacity = alignedSize(requested);
    std::scoped_lock lock(state->mutex);
    if (!state->accepting) throw std::runtime_error("CUDA memory pool is shut down");
    const auto reusable = state->freeBlocks.lower_bound(capacity);
    void* pointer = nullptr;
    std::size_t actualCapacity = capacity;
    bool wasReused = false;
    if (reusable != state->freeBlocks.end()) {
        wasReused = true;
        actualCapacity = reusable->first;
        pointer = reusable->second;
        state->freeBlocks.erase(reusable);
        state->statistics.freeBytes -= actualCapacity;
        ++state->statistics.reuseCount;
    } else {
        pointer = state->backend->allocate(capacity);
        state->statistics.allocatedBytes += capacity;
        ++state->statistics.allocationCount;
    }
    try {
        if (!state->activeBlocks.emplace(pointer, actualCapacity).second) {
            throw std::logic_error("memory-pool pointer is already active");
        }
    } catch (...) {
        state->backend->free(pointer);
        state->statistics.allocatedBytes -= actualCapacity;
        if (wasReused) --state->statistics.reuseCount;
        else --state->statistics.allocationCount;
        throw;
    }
    state->statistics.activeBytes += actualCapacity;
    state->statistics.peakUsageBytes =
        std::max(state->statistics.peakUsageBytes, state->statistics.activeBytes);
    return {pointer, actualCapacity};
}

void release(const std::shared_ptr<CudaBuffer::State>& state, void* pointer) noexcept {
    if (!state || pointer == nullptr) return;
    std::size_t capacity = 0;
    bool cache = false;
    {
        std::scoped_lock lock(state->mutex);
        const auto active = state->activeBlocks.find(pointer);
        if (active == state->activeBlocks.end()) return;
        capacity = active->second;
        state->activeBlocks.erase(active);
        state->statistics.activeBytes -= capacity;
        const auto freeBytes = static_cast<std::size_t>(state->statistics.freeBytes);
        cache = state->accepting && freeBytes <= state->maximumCachedBytes &&
                capacity <= state->maximumCachedBytes - freeBytes;
        if (cache) {
            try {
                state->freeBlocks.emplace(capacity, pointer);
                state->statistics.freeBytes += capacity;
                return;
            } catch (...) {
                cache = false;
            }
        }
        state->statistics.allocatedBytes -= capacity;
    }
    state->backend->free(pointer);
}

void drainFreeBlocks(const std::shared_ptr<CudaBuffer::State>& state) noexcept {
    while (true) {
        void* pointer = nullptr;
        {
            std::scoped_lock lock(state->mutex);
            if (state->freeBlocks.empty()) break;
            const auto block = state->freeBlocks.begin();
            state->statistics.allocatedBytes -= block->first;
            state->statistics.freeBytes -= block->first;
            pointer = block->second;
            state->freeBlocks.erase(block);
        }
        state->backend->free(pointer);
    }
}

} // namespace

CudaBuffer::CudaBuffer(std::shared_ptr<State> state,
                       void* data,
                       std::size_t size,
                       std::size_t capacity) noexcept
    : state_(std::move(state)), data_(data), size_(size), capacity_(capacity) {}

CudaBuffer::~CudaBuffer() { reset(); }

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : state_(std::move(other.state_)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      capacity_(std::exchange(other.capacity_, 0)) {}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
}

void CudaBuffer::reset() noexcept {
    release(state_, data_);
    state_.reset();
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
}

void* CudaBuffer::data() noexcept { return data_; }
const void* CudaBuffer::data() const noexcept { return data_; }
std::size_t CudaBuffer::size() const noexcept { return size_; }
std::size_t CudaBuffer::capacity() const noexcept { return capacity_; }
CudaBuffer::operator bool() const noexcept { return data_ != nullptr; }

CudaMemoryPool::CudaMemoryPool(std::shared_ptr<ComputeBackend> backend,
                               std::size_t maximumCachedBytes)
    : state_(std::make_shared<CudaBuffer::State>(std::move(backend),
                                                 maximumCachedBytes)) {
    if (!state_->backend || !state_->backend->isAvailable()) {
        throw std::invalid_argument("CudaMemoryPool requires an available backend");
    }
}

CudaMemoryPool::~CudaMemoryPool() { shutdown(); }

CudaBuffer CudaMemoryPool::allocate(std::size_t sizeBytes) {
    const auto allocation = acquire(state_, sizeBytes);
    return {state_, allocation.pointer, sizeBytes, allocation.capacity};
}

std::shared_ptr<DeviceBuffer>
CudaMemoryPool::allocateDeviceBuffer(std::size_t sizeBytes) {
    const auto allocation = acquire(state_, sizeBytes);
    try {
        auto state = state_;
        return std::make_shared<DeviceBuffer>(
            state_->backend, allocation.pointer, sizeBytes,
            [state = std::move(state)](void* pointer) noexcept {
                release(state, pointer);
            });
    } catch (...) {
        release(state_, allocation.pointer);
        throw;
    }
}

void CudaMemoryPool::trim() noexcept { drainFreeBlocks(state_); }

void CudaMemoryPool::shutdown() noexcept {
    if (!state_) return;
    {
        std::scoped_lock lock(state_->mutex);
        state_->accepting = false;
    }
    drainFreeBlocks(state_);
}

CudaMemoryPoolStats CudaMemoryPool::stats() const {
    std::scoped_lock lock(state_->mutex);
    return state_->statistics;
}

const std::shared_ptr<ComputeBackend>& CudaMemoryPool::backend() const noexcept {
    return state_->backend;
}

} // namespace hypermoe::backend
