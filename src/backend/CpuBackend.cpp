#include "backend/CpuBackend.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>

namespace hypermoe::backend {
namespace {
constexpr std::align_val_t kAlignment{64};
}

CpuBackend::~CpuBackend() {
    std::scoped_lock lock(mutex_);
    for (const auto& [pointer, size] : allocations_) {
        (void)size;
        ::operator delete(pointer, kAlignment);
    }
}

BackendKind CpuBackend::kind() const noexcept { return BackendKind::Cpu; }
std::string_view CpuBackend::name() const noexcept { return "CPU"; }
int CpuBackend::deviceOrdinal() const noexcept { return 0; }
bool CpuBackend::isAvailable() const noexcept { return true; }

void* CpuBackend::allocate(std::size_t sizeBytes) {
    if (sizeBytes == 0) {
        throw std::invalid_argument("CPU allocation size must be nonzero");
    }
    void* pointer = ::operator new(sizeBytes, kAlignment);
    try {
        std::scoped_lock lock(mutex_);
        allocations_.emplace(pointer, sizeBytes);
        stats_.allocatedBytes += sizeBytes;
        stats_.peakAllocatedBytes =
            std::max(stats_.peakAllocatedBytes, stats_.allocatedBytes);
    } catch (...) {
        ::operator delete(pointer, kAlignment);
        throw;
    }
    return pointer;
}

void CpuBackend::free(void* pointer) noexcept {
    if (pointer == nullptr) return;
    std::size_t size = 0;
    {
        std::scoped_lock lock(mutex_);
        const auto it = allocations_.find(pointer);
        if (it == allocations_.end()) return;
        size = it->second;
        stats_.allocatedBytes -= size;
        allocations_.erase(it);
    }
    ::operator delete(pointer, kAlignment);
}

void CpuBackend::copyToDevice(void* destination,
                              const void* source,
                              std::size_t sizeBytes,
                              StreamHandle) {
    copy(destination, source, sizeBytes, true);
}

void CpuBackend::copyFromDevice(void* destination,
                                const void* source,
                                std::size_t sizeBytes,
                                StreamHandle) {
    copy(destination, source, sizeBytes, false);
}

void CpuBackend::synchronize(StreamHandle) {}

MemoryInfo CpuBackend::getMemoryInfo() const {
    std::scoped_lock lock(mutex_);
    return {0, 0, stats_.allocatedBytes};
}

BackendStats CpuBackend::stats() const {
    std::scoped_lock lock(mutex_);
    return stats_;
}

StreamHandle CpuBackend::createStream() { return nullptr; }
void CpuBackend::destroyStream(StreamHandle) noexcept {}
EventHandle CpuBackend::createEvent() { return nullptr; }
void CpuBackend::recordEvent(EventHandle, StreamHandle) {}
void CpuBackend::waitEvent(EventHandle) {}
void CpuBackend::destroyEvent(EventHandle) noexcept {}

void* CpuBackend::allocatePinned(std::size_t sizeBytes) {
    return allocate(sizeBytes);
}

void CpuBackend::freePinned(void* pointer) noexcept { free(pointer); }
bool CpuBackend::supportsPinnedMemory() const noexcept { return false; }

void CpuBackend::copy(void* destination,
                      const void* source,
                      std::size_t sizeBytes,
                      bool toDevice) {
    if ((destination == nullptr || source == nullptr) && sizeBytes != 0) {
        throw std::invalid_argument("CPU copy received a null pointer");
    }
    const auto start = std::chrono::steady_clock::now();
    std::memmove(destination, source, sizeBytes);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    std::scoped_lock lock(mutex_);
    if (toDevice) stats_.hostToDeviceBytes += sizeBytes;
    else stats_.deviceToHostBytes += sizeBytes;
    stats_.transferTime += elapsed;
}

} // namespace hypermoe::backend
