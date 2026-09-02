#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace hypermoe::backend {

using StreamHandle = void*;
using EventHandle = void*;

enum class BackendKind {
    Cpu,
    Cuda,
};

struct MemoryInfo {
    std::uint64_t totalBytes{};
    std::uint64_t freeBytes{};
    std::uint64_t allocatedBytes{};
};

struct BackendStats {
    std::uint64_t allocatedBytes{};
    std::uint64_t peakAllocatedBytes{};
    std::uint64_t hostToDeviceBytes{};
    std::uint64_t deviceToHostBytes{};
    std::chrono::nanoseconds transferTime{};
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    [[nodiscard]] virtual BackendKind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool isAvailable() const noexcept = 0;

    [[nodiscard]] virtual void* allocate(std::size_t sizeBytes) = 0;
    virtual void free(void* pointer) noexcept = 0;

    virtual void copyToDevice(void* destination,
                              const void* source,
                              std::size_t sizeBytes,
                              StreamHandle stream = nullptr) = 0;
    virtual void copyFromDevice(void* destination,
                                const void* source,
                                std::size_t sizeBytes,
                                StreamHandle stream = nullptr) = 0;
    virtual void synchronize(StreamHandle stream = nullptr) = 0;

    [[nodiscard]] virtual MemoryInfo getMemoryInfo() const = 0;
    [[nodiscard]] virtual BackendStats stats() const = 0;

    [[nodiscard]] virtual StreamHandle createStream() = 0;
    virtual void destroyStream(StreamHandle stream) noexcept = 0;
    [[nodiscard]] virtual EventHandle createEvent() = 0;
    virtual void recordEvent(EventHandle event, StreamHandle stream) = 0;
    virtual void waitEvent(EventHandle event) = 0;
    virtual void destroyEvent(EventHandle event) noexcept = 0;

    [[nodiscard]] virtual void* allocatePinned(std::size_t sizeBytes) = 0;
    virtual void freePinned(void* pointer) noexcept = 0;
    [[nodiscard]] virtual bool supportsPinnedMemory() const noexcept = 0;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(std::shared_ptr<ComputeBackend> backend, std::size_t sizeBytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void reset() noexcept;
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const std::shared_ptr<ComputeBackend>& backend() const noexcept;

private:
    std::shared_ptr<ComputeBackend> backend_;
    void* data_{};
    std::size_t size_{};
};

} // namespace hypermoe::backend
