#include "backend/CudaBackend.hpp"

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

cudaEvent_t eventFrom(EventHandle event) {
    return reinterpret_cast<cudaEvent_t>(event);
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
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    impl_->deviceName = properties.name;
#else
    (void)device;
    throw std::runtime_error("HyperMoE was built without CUDA support");
#endif
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
#ifdef HYPERMOE_HAS_CUDA
    return true;
#else
    return false;
#endif
}

CudaRuntimeInfo CudaBackend::query(int device) noexcept {
    CudaRuntimeInfo info;
#ifdef HYPERMOE_HAS_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) return info;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return info;
    info.available = true;
    info.deviceName = properties.name;
    info.totalVramBytes = properties.totalGlobalMem;
    (void)cudaRuntimeGetVersion(&info.runtimeVersion);
    (void)cudaDriverGetVersion(&info.driverVersion);
#else
    (void)device;
#endif
    return info;
}

BackendKind CudaBackend::kind() const noexcept { return BackendKind::Cuda; }
std::string_view CudaBackend::name() const noexcept { return impl_->deviceName; }
bool CudaBackend::isAvailable() const noexcept { return query(impl_->device).available; }

void* CudaBackend::allocate(std::size_t sizeBytes) {
    if (sizeBytes == 0) throw std::invalid_argument("CUDA allocation size must be nonzero");
#ifdef HYPERMOE_HAS_CUDA
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
#ifdef HYPERMOE_HAS_CUDA
    cudaStream_t stream{};
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    return reinterpret_cast<StreamHandle>(stream);
#else
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::destroyStream(StreamHandle stream) noexcept {
#ifdef HYPERMOE_HAS_CUDA
    if (stream != nullptr) (void)cudaStreamDestroy(streamFrom(stream));
#else
    (void)stream;
#endif
}

EventHandle CudaBackend::createEvent() {
#ifdef HYPERMOE_HAS_CUDA
    cudaEvent_t event{};
    checkCuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
    return reinterpret_cast<EventHandle>(event);
#else
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::recordEvent(EventHandle event, StreamHandle stream) {
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaEventRecord(eventFrom(event), streamFrom(stream)), "cudaEventRecord");
#else
    (void)event; (void)stream;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::waitEvent(EventHandle event) {
#ifdef HYPERMOE_HAS_CUDA
    checkCuda(cudaEventSynchronize(eventFrom(event)), "cudaEventSynchronize");
#else
    (void)event;
    throw std::runtime_error("CUDA support is unavailable");
#endif
}

void CudaBackend::destroyEvent(EventHandle event) noexcept {
#ifdef HYPERMOE_HAS_CUDA
    if (event != nullptr) (void)cudaEventDestroy(eventFrom(event));
#else
    (void)event;
#endif
}

void* CudaBackend::allocatePinned(std::size_t sizeBytes) {
    if (sizeBytes == 0) throw std::invalid_argument("pinned allocation size must be nonzero");
#ifdef HYPERMOE_HAS_CUDA
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
