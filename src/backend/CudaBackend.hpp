#pragma once

#include "backend/Backend.hpp"

#include <memory>
#include <string>

namespace hypermoe::backend {

struct CudaRuntimeInfo {
    bool available{};
    std::string deviceName;
    int computeCapabilityMajor{};
    int computeCapabilityMinor{};
    std::uint64_t totalVramBytes{};
    std::uint64_t freeVramBytes{};
    int runtimeVersion{};
    int driverVersion{};
};

class CudaBackend final : public ComputeBackend {
public:
    struct Impl;

    explicit CudaBackend(int device = 0);
    ~CudaBackend() override;

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    [[nodiscard]] static bool compiledWithCuda() noexcept;
    [[nodiscard]] static CudaRuntimeInfo query(int device = 0);

    [[nodiscard]] BackendKind kind() const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
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
    std::unique_ptr<Impl> impl_;
};

} // namespace hypermoe::backend
