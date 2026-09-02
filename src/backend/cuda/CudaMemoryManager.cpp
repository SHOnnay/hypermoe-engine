#include "backend/CudaBackend.hpp"
#include "backend/cuda/CudaRuntime.hpp"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef HYPERMOE_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace hypermoe::backend {

struct CudaBackend::Impl {
    int device{};
    std::string deviceName{"CUDA unavailable"};
    std::shared_ptr<CudaRuntime> runtime;
    mutable std::mutex mutex;
    std::unordered_map<void*, std::size_t> allocations;
    std::unordered_map<void*, std::size_t> pinnedAllocations;
    BackendStats statistics;
#ifdef HYPERMOE_HAS_CUDA
    struct Timing {
        cudaEvent_t start{};
        cudaEvent_t end{};
        cudaStream_t stream{};
    };
    std::vector<Timing> pendingTimings;
#endif
};

#ifdef HYPERMOE_HAS_CUDA
namespace {

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

cudaStream_t streamFrom(StreamHandle stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

void collectTimings(CudaBackend::Impl& impl, cudaStream_t selected, bool all) {
    std::scoped_lock lock(impl.mutex);
    auto current = impl.pendingTimings.begin();
    while (current != impl.pendingTimings.end()) {
        if (!all && current->stream != selected) {
            ++current;
            continue;
        }
        float milliseconds = 0.0F;
        if (cudaEventElapsedTime(&milliseconds, current->start, current->end) == cudaSuccess) {
            impl.statistics.transferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double, std::milli>(milliseconds));
        }
        (void)cudaEventDestroy(current->start);
        (void)cudaEventDestroy(current->end);
        current = impl.pendingTimings.erase(current);
    }
}

} // namespace
#endif

CudaBackend::CudaBackend(int device) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    impl_->runtime = std::make_shared<CudaRuntime>(device);
    const auto info = impl_->runtime->deviceInfo();
    if (!info.available) throw std::runtime_error("CUDA runtime/device is unavailable");
    impl_->deviceName = info.name;
}

CudaBackend::~CudaBackend() {
#ifdef HYPERMOE_HAS_CUDA
    if (!impl_) return;
    (void)cudaSetDevice(impl_->device);
    (void)cudaDeviceSynchronize();
    collectTimings(*impl_, nullptr, true);
    for (const auto& [pointer, size] : impl_->allocations) {
        (void)size;
        (void)cudaFree(pointer);
    }
    for (const auto& [pointer, size] : impl_->pinnedAllocations) {
        (void)size;
        (void)cudaFreeHost(pointer);
    }
#endif
}

bool CudaBackend::compiledWithCuda() noexcept {
    return CudaRuntime::compiledWithCuda();
}

CudaRuntimeInfo CudaBackend::query(int device) {
    CudaRuntimeInfo info;
    const auto deviceInfo = CudaRuntime::query(device);
    info.available = deviceInfo.available;
    info.deviceName = deviceInfo.name;
    info.computeCapabilityMajor = deviceInfo.computeCapabilityMajor;
    info.computeCapabilityMinor = deviceInfo.computeCapabilityMinor;
    info.totalVramBytes = deviceInfo.totalVramBytes;
    info.freeVramBytes = deviceInfo.freeVramBytes;
    info.runtimeVersion = deviceInfo.runtimeVersion;
    info.driverVersion = deviceInfo.driverVersion;
    return info;
}

BackendKind CudaBackend::kind() const noexcept { return BackendKind::Cuda; }
std::string_view CudaBackend::name() const noexcept { return impl_->deviceName; }
int CudaBackend::deviceOrdinal() const noexcept { return impl_->device; }
bool CudaBackend::isAvailable() const noexcept {
    return impl_->runtime && impl_->runtime->available();
}

void* CudaBackend::allocate(std::size_t sizeBytes) {
    if (sizeBytes == 0) throw std::invalid_argument("CUDA allocation size must be nonzero");
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    void* pointer = nullptr;
    checkCuda(cudaMalloc(&pointer, sizeBytes), "cudaMalloc");
    try {
        std::scoped_lock lock(impl_->mutex);
        impl_->allocations.emplace(pointer, sizeBytes);
        impl_->statistics.allocatedBytes += sizeBytes;
        impl_->statistics.peakAllocatedBytes =
            std::max(impl_->statistics.peakAllocatedBytes,
                     impl_->statistics.allocatedBytes);
    } catch (...) {
        (void)cudaFree(pointer);
        throw;
    }
    return pointer;
#else
    (void)sizeBytes;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::free(void* pointer) noexcept {
#ifdef HYPERMOE_HAS_CUDA
    if (pointer == nullptr) return;
    if (cudaSetDevice(impl_->device) != cudaSuccess) return;
    std::scoped_lock lock(impl_->mutex);
    const auto it = impl_->allocations.find(pointer);
    if (it == impl_->allocations.end()) return;
    if (cudaFree(pointer) == cudaSuccess) {
        impl_->statistics.allocatedBytes -= it->second;
        impl_->allocations.erase(it);
    }
#else
    (void)pointer;
#endif
}

void CudaBackend::copyToDevice(void* destination,
                               const void* source,
                               std::size_t sizeBytes,
                               StreamHandle stream) {
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    cudaEvent_t start{};
    cudaEvent_t end{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    try {
        checkCuda(cudaEventCreate(&end), "cudaEventCreate(end)");
        checkCuda(cudaEventRecord(start, streamFrom(stream)), "cudaEventRecord(start)");
        checkCuda(cudaMemcpyAsync(destination, source, sizeBytes, cudaMemcpyHostToDevice,
                                  streamFrom(stream)),
                  "cudaMemcpyAsync(host-to-device)");
        checkCuda(cudaEventRecord(end, streamFrom(stream)), "cudaEventRecord(end)");
        std::scoped_lock lock(impl_->mutex);
        impl_->statistics.hostToDeviceBytes += sizeBytes;
        impl_->pendingTimings.push_back({start, end, streamFrom(stream)});
    } catch (...) {
        (void)cudaEventDestroy(start);
        if (end != nullptr) (void)cudaEventDestroy(end);
        throw;
    }
#else
    (void)destination; (void)source; (void)sizeBytes; (void)stream;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::copyFromDevice(void* destination,
                                 const void* source,
                                 std::size_t sizeBytes,
                                 StreamHandle stream) {
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    cudaEvent_t start{};
    cudaEvent_t end{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    try {
        checkCuda(cudaEventCreate(&end), "cudaEventCreate(end)");
        checkCuda(cudaEventRecord(start, streamFrom(stream)), "cudaEventRecord(start)");
        checkCuda(cudaMemcpyAsync(destination, source, sizeBytes, cudaMemcpyDeviceToHost,
                                  streamFrom(stream)),
                  "cudaMemcpyAsync(device-to-host)");
        checkCuda(cudaEventRecord(end, streamFrom(stream)), "cudaEventRecord(end)");
        std::scoped_lock lock(impl_->mutex);
        impl_->statistics.deviceToHostBytes += sizeBytes;
        impl_->pendingTimings.push_back({start, end, streamFrom(stream)});
    } catch (...) {
        (void)cudaEventDestroy(start);
        if (end != nullptr) (void)cudaEventDestroy(end);
        throw;
    }
#else
    (void)destination; (void)source; (void)sizeBytes; (void)stream;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::synchronize(StreamHandle stream) {
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    if (stream == nullptr) {
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        collectTimings(*impl_, nullptr, true);
    } else {
        checkCuda(cudaStreamSynchronize(streamFrom(stream)), "cudaStreamSynchronize");
        collectTimings(*impl_, streamFrom(stream), false);
    }
#else
    (void)stream;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

MemoryInfo CudaBackend::getMemoryInfo() const {
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
    std::scoped_lock lock(impl_->mutex);
    return {totalBytes, freeBytes, impl_->statistics.allocatedBytes};
#else
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

BackendStats CudaBackend::stats() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->statistics;
}

StreamHandle CudaBackend::createStream() {
    return impl_->runtime->createStream();
}

void CudaBackend::destroyStream(StreamHandle stream) noexcept {
    impl_->runtime->destroyStream(stream);
}

EventHandle CudaBackend::createEvent() {
    return impl_->runtime->createEvent(false);
}

void CudaBackend::recordEvent(EventHandle event, StreamHandle stream) {
    impl_->runtime->recordEvent(event, stream);
}

void CudaBackend::waitEvent(EventHandle event) {
    impl_->runtime->synchronizeEvent(event);
}

void CudaBackend::destroyEvent(EventHandle event) noexcept {
    impl_->runtime->destroyEvent(event);
}

void* CudaBackend::allocatePinned(std::size_t sizeBytes) {
    if (sizeBytes == 0) throw std::invalid_argument("pinned allocation size must be nonzero");
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    void* pointer = nullptr;
    checkCuda(cudaHostAlloc(&pointer, sizeBytes, cudaHostAllocPortable), "cudaHostAlloc");
    try {
        std::scoped_lock lock(impl_->mutex);
        impl_->pinnedAllocations.emplace(pointer, sizeBytes);
    } catch (...) {
        (void)cudaFreeHost(pointer);
        throw;
    }
    return pointer;
#else
    (void)sizeBytes;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::freePinned(void* pointer) noexcept {
#ifdef HYPERMOE_HAS_CUDA
    if (pointer == nullptr) return;
    std::scoped_lock lock(impl_->mutex);
    const auto it = impl_->pinnedAllocations.find(pointer);
    if (it == impl_->pinnedAllocations.end()) return;
    if (cudaFreeHost(pointer) == cudaSuccess) impl_->pinnedAllocations.erase(it);
#else
    (void)pointer;
#endif
}

bool CudaBackend::supportsPinnedMemory() const noexcept {
    return isAvailable();
}

} // namespace hypermoe::backend
