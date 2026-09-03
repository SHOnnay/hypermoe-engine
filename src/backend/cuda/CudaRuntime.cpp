#include "backend/cuda/CudaRuntime.hpp"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#ifdef HYPERMOE_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace hypermoe::backend {

struct CudaRuntime::Impl {
    int device{};
    std::atomic_bool available{};
    mutable std::mutex mutex;
    std::unordered_set<StreamHandle> streams;
    std::unordered_set<EventHandle> events;
};

#ifdef HYPERMOE_HAS_CUDA
namespace {

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

cudaStream_t nativeStream(StreamHandle stream) noexcept {
    return reinterpret_cast<cudaStream_t>(stream);
}

cudaEvent_t nativeEvent(EventHandle event) noexcept {
    return reinterpret_cast<cudaEvent_t>(event);
}

} // namespace
#endif

CudaRuntime::CudaRuntime(int device) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
#ifdef HYPERMOE_HAS_CUDA
    int count = 0;
    if (device < 0 || cudaGetDeviceCount(&count) != cudaSuccess || device >= count) return;
    if (cudaSetDevice(device) != cudaSuccess) return;
    cudaDeviceProp properties{};
    impl_->available.store(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
                           std::memory_order_relaxed);
#else
    (void)device;
#endif
}

CudaRuntime::~CudaRuntime() { shutdown(); }

bool CudaRuntime::compiledWithCuda() noexcept {
#ifdef HYPERMOE_HAS_CUDA
    return true;
#else
    return false;
#endif
}

DeviceInfo CudaRuntime::query(int device) {
    DeviceInfo info;
    info.deviceOrdinal = device;
#ifdef HYPERMOE_HAS_CUDA
    int count = 0;
    if (device < 0 || cudaGetDeviceCount(&count) != cudaSuccess || device >= count) {
        return info;
    }
    if (cudaSetDevice(device) != cudaSuccess) return info;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return info;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) return info;
    info.available = true;
    info.name = properties.name;
    info.computeCapabilityMajor = properties.major;
    info.computeCapabilityMinor = properties.minor;
    info.totalVramBytes = totalBytes;
    info.freeVramBytes = freeBytes;
    (void)cudaRuntimeGetVersion(&info.runtimeVersion);
    (void)cudaDriverGetVersion(&info.driverVersion);
#else
    (void)device;
#endif
    return info;
}

bool CudaRuntime::available() const noexcept {
    return impl_ && impl_->available.load(std::memory_order_relaxed);
}
int CudaRuntime::deviceOrdinal() const noexcept { return impl_ ? impl_->device : -1; }

DeviceInfo CudaRuntime::deviceInfo() const {
    auto info = query(deviceOrdinal());
    if (impl_) {
        std::scoped_lock lock(impl_->mutex);
        info.streamCount = static_cast<std::uint32_t>(impl_->streams.size());
    }
    return info;
}

StreamHandle CudaRuntime::createStream() {
    if (!available()) throw std::runtime_error("CUDA runtime is unavailable");
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    cudaStream_t stream{};
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    const auto handle = reinterpret_cast<StreamHandle>(stream);
    try {
        std::scoped_lock lock(impl_->mutex);
        impl_->streams.insert(handle);
    } catch (...) {
        (void)cudaStreamDestroy(stream);
        throw;
    }
    return handle;
#else
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaRuntime::destroyStream(StreamHandle stream) noexcept {
    if (!impl_ || stream == nullptr) return;
#ifdef HYPERMOE_HAS_CUDA
    {
        std::scoped_lock lock(impl_->mutex);
        if (impl_->streams.erase(stream) == 0) return;
    }
    (void)cudaSetDevice(impl_->device);
    (void)cudaStreamDestroy(nativeStream(stream));
#else
    (void)stream;
#endif
}

EventHandle CudaRuntime::createEvent(bool enableTiming) {
    if (!available()) throw std::runtime_error("CUDA runtime is unavailable");
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    cudaEvent_t event{};
    const auto flags = enableTiming ? cudaEventDefault : cudaEventDisableTiming;
    checkCuda(cudaEventCreateWithFlags(&event, flags), "cudaEventCreateWithFlags");
    const auto handle = reinterpret_cast<EventHandle>(event);
    try {
        std::scoped_lock lock(impl_->mutex);
        impl_->events.insert(handle);
    } catch (...) {
        (void)cudaEventDestroy(event);
        throw;
    }
    return handle;
#else
    (void)enableTiming;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaRuntime::recordEvent(EventHandle event, StreamHandle stream) {
    if (!available() || event == nullptr) {
        throw std::invalid_argument("CUDA event is unavailable");
    }
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    checkCuda(cudaEventRecord(nativeEvent(event), nativeStream(stream)),
              "cudaEventRecord");
#else
    (void)stream;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaRuntime::synchronizeEvent(EventHandle event) {
    if (!available() || event == nullptr) {
        throw std::invalid_argument("CUDA event is unavailable");
    }
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    checkCuda(cudaEventSynchronize(nativeEvent(event)), "cudaEventSynchronize");
#else
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

float CudaRuntime::elapsedMilliseconds(EventHandle start, EventHandle end) const {
    if (!available() || start == nullptr || end == nullptr) {
        throw std::invalid_argument("timed CUDA events are unavailable");
    }
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    float milliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&milliseconds, nativeEvent(start), nativeEvent(end)),
              "cudaEventElapsedTime");
    return milliseconds;
#else
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaRuntime::destroyEvent(EventHandle event) noexcept {
    if (!impl_ || event == nullptr) return;
#ifdef HYPERMOE_HAS_CUDA
    {
        std::scoped_lock lock(impl_->mutex);
        if (impl_->events.erase(event) == 0) return;
    }
    (void)cudaSetDevice(impl_->device);
    (void)cudaEventDestroy(nativeEvent(event));
#else
    (void)event;
#endif
}

void CudaRuntime::synchronize(StreamHandle stream) {
    if (!available()) throw std::runtime_error("CUDA runtime is unavailable");
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    if (stream == nullptr) checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    else checkCuda(cudaStreamSynchronize(nativeStream(stream)), "cudaStreamSynchronize");
#else
    (void)stream;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaRuntime::shutdown() noexcept {
    if (!impl_) return;
#ifdef HYPERMOE_HAS_CUDA
    std::unordered_set<EventHandle> events;
    std::unordered_set<StreamHandle> streams;
    {
        std::scoped_lock lock(impl_->mutex);
        if (!impl_->available.load(std::memory_order_relaxed) && impl_->events.empty() &&
            impl_->streams.empty()) {
            return;
        }
        impl_->available.store(false, std::memory_order_relaxed);
        events.swap(impl_->events);
        streams.swap(impl_->streams);
    }
    (void)cudaSetDevice(impl_->device);
    (void)cudaDeviceSynchronize();
    for (const auto event : events) (void)cudaEventDestroy(nativeEvent(event));
    for (const auto stream : streams) (void)cudaStreamDestroy(nativeStream(stream));
#endif
}

} // namespace hypermoe::backend
