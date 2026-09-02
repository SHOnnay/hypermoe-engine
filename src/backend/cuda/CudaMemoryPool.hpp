#pragma once

#include "backend/Backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace hypermoe::backend {

struct CudaMemoryPoolStats {
    std::uint64_t allocatedBytes{};
    std::uint64_t activeBytes{};
    std::uint64_t freeBytes{};
    std::uint64_t peakUsageBytes{};
    std::uint64_t allocationCount{};
    std::uint64_t reuseCount{};
};

class CudaBuffer {
public:
    struct State;

    CudaBuffer() = default;
    ~CudaBuffer();

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    void reset() noexcept;
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    friend class CudaMemoryPool;

    CudaBuffer(std::shared_ptr<State> state,
               void* data,
               std::size_t size,
               std::size_t capacity) noexcept;

    std::shared_ptr<State> state_;
    void* data_{};
    std::size_t size_{};
    std::size_t capacity_{};
};

class CudaMemoryPool {
public:
    explicit CudaMemoryPool(std::shared_ptr<ComputeBackend> backend,
                            std::size_t maximumCachedBytes = 512ULL * 1024ULL * 1024ULL);
    ~CudaMemoryPool();

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    [[nodiscard]] CudaBuffer allocate(std::size_t sizeBytes);
    [[nodiscard]] std::shared_ptr<DeviceBuffer>
    allocateDeviceBuffer(std::size_t sizeBytes);
    void trim() noexcept;
    void shutdown() noexcept;
    [[nodiscard]] CudaMemoryPoolStats stats() const;
    [[nodiscard]] const std::shared_ptr<ComputeBackend>& backend() const noexcept;

private:
    std::shared_ptr<CudaBuffer::State> state_;
};

} // namespace hypermoe::backend
