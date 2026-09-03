#pragma once

#include "backend/Backend.hpp"
#include "backend/cuda/CudaDeviceInfo.hpp"

#include <memory>

namespace hypermoe::backend {

class CudaRuntime {
public:
    struct Impl;

    explicit CudaRuntime(int device = 0);
    ~CudaRuntime();

    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;

    [[nodiscard]] static bool compiledWithCuda() noexcept;
    [[nodiscard]] static DeviceInfo query(int device = 0);

    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] int deviceOrdinal() const noexcept;
    [[nodiscard]] DeviceInfo deviceInfo() const;

    [[nodiscard]] StreamHandle createStream();
    void destroyStream(StreamHandle stream) noexcept;
    [[nodiscard]] EventHandle createEvent(bool enableTiming = false);
    void recordEvent(EventHandle event, StreamHandle stream = nullptr);
    void synchronizeEvent(EventHandle event);
    [[nodiscard]] float elapsedMilliseconds(EventHandle start,
                                            EventHandle end) const;
    void destroyEvent(EventHandle event) noexcept;
    void synchronize(StreamHandle stream = nullptr);
    void shutdown() noexcept;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace hypermoe::backend
