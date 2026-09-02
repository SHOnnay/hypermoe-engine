#pragma once

#include "backend/Backend.hpp"

#include <mutex>
#include <unordered_map>

namespace hypermoe::backend {

class CpuBackend final : public ComputeBackend {
public:
    CpuBackend() = default;
    ~CpuBackend() override;

    [[nodiscard]] BackendKind kind() const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] int deviceOrdinal() const noexcept override;
    [[nodiscard]] bool isAvailable() const noexcept override;
    [[nodiscard]] void* allocate(std::size_t sizeBytes) override;
    void free(void* pointer) noexcept override;
    void copyToDevice(void* destination,
                      const void* source,
                      std::size_t sizeBytes,
                      StreamHandle stream = nullptr) override;
    void copyFromDevice(void* destination,
                        const void* source,
                        std::size_t sizeBytes,
                        StreamHandle stream = nullptr) override;
    void synchronize(StreamHandle stream = nullptr) override;
    [[nodiscard]] MemoryInfo getMemoryInfo() const override;
    [[nodiscard]] BackendStats stats() const override;
    [[nodiscard]] StreamHandle createStream() override;
    void destroyStream(StreamHandle stream) noexcept override;
    [[nodiscard]] EventHandle createEvent() override;
    void recordEvent(EventHandle event, StreamHandle stream) override;
    void waitEvent(EventHandle event) override;
    void destroyEvent(EventHandle event) noexcept override;
    [[nodiscard]] void* allocatePinned(std::size_t sizeBytes) override;
    void freePinned(void* pointer) noexcept override;
    [[nodiscard]] bool supportsPinnedMemory() const noexcept override;

private:
    void copy(void* destination, const void* source, std::size_t sizeBytes, bool toDevice);

    mutable std::mutex mutex_;
    std::unordered_map<void*, std::size_t> allocations_;
    BackendStats stats_;
};

} // namespace hypermoe::backend
