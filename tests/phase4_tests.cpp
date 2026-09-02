#include "backend/CpuBackend.hpp"
#include "backend/CudaBackend.hpp"
#include "backend/cuda/CudaMemoryPool.hpp"
#include "backend/cuda/CudaRuntime.hpp"
#include "backend/cuda/CudaStreamManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "memory/PinnedBuffer.hpp"
#include "scheduler/RuntimeEvent.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void testRuntimeFallbackAndCapabilities() {
    hypermoe::backend::CudaRuntime runtime;
    const auto queried = hypermoe::backend::CudaRuntime::query();
    expect(runtime.available() == queried.available,
           "runtime availability agrees with static capability query");
    expect(hypermoe::backend::CudaRuntime::compiledWithCuda() ==
               hypermoe::backend::CudaBackend::compiledWithCuda(),
           "runtime and backend expose the same CUDA compile state");
    if (!runtime.available()) {
        expect(runtime.deviceInfo().streamCount == 0,
               "unavailable CUDA runtime owns no streams");
        try {
            (void)runtime.createStream();
            expect(false, "unavailable runtime rejects stream creation");
        } catch (const std::runtime_error&) {
            expect(true, "unavailable runtime rejects stream creation");
        }
    }

    const auto hardware = hypermoe::hardware::detectHardware();
    expect(hardware.cudaAvailable == queried.available,
           "hardware detection uses the CUDA runtime capability result");
    expect(hardware.toJson().find("\"compute_capability\"") != std::string::npos &&
               hardware.toJson().find("\"free_vram_bytes\"") != std::string::npos,
           "hardware JSON includes Phase 4 device capabilities");
}

void testMemoryPoolWithCpuBackend() {
    auto backend = std::make_shared<hypermoe::backend::CpuBackend>();
    hypermoe::backend::CudaMemoryPool pool(backend, 4096);
    void* firstPointer = nullptr;
    {
        auto first = pool.allocate(1000);
        firstPointer = first.data();
        expect(first.size() == 1000 && first.capacity() == 1024,
               "memory pool aligns capacity while preserving logical size");
        std::fill_n(static_cast<std::byte*>(first.data()), first.size(),
                    std::byte{0x4a});
        const auto active = pool.stats();
        expect(active.activeBytes == 1024 && active.peakUsageBytes == 1024 &&
                   active.allocationCount == 1,
               "memory pool accounts for an active physical allocation");
    }
    const auto cached = pool.stats();
    expect(cached.activeBytes == 0 && cached.freeBytes == 1024,
           "RAII release returns a block to the pool");
    {
        auto reused = pool.allocate(768);
        expect(reused.data() == firstPointer && pool.stats().reuseCount == 1,
               "best-fit allocation reuses a cached block");
    }

    {
        auto adopted = pool.allocateDeviceBuffer(513);
        expect(adopted && adopted->size() == 513,
               "pool can provide DeviceBuffer-compatible RAII ownership");
    }
    expect(pool.stats().reuseCount == 2,
           "DeviceBuffer-compatible allocation also reuses the cache");
    pool.trim();
    expect(pool.stats().allocatedBytes == 0 &&
               backend->stats().allocatedBytes == 0,
           "pool trimming releases cached blocks to the backend");
}

void testPoolBufferOutlivesPool() {
    auto backend = std::make_shared<hypermoe::backend::CpuBackend>();
    hypermoe::backend::CudaBuffer buffer;
    {
        hypermoe::backend::CudaMemoryPool pool(backend, 4096);
        buffer = pool.allocate(2048);
    }
    expect(backend->stats().allocatedBytes == 2048,
           "active pooled allocation remains valid after pool shutdown");
    buffer.reset();
    expect(backend->stats().allocatedBytes == 0,
           "outliving buffer releases directly when its pool is gone");
}

void testStreamRolesAndRuntimeEvents() {
    auto runtime = std::make_shared<hypermoe::backend::CudaRuntime>();
    hypermoe::backend::CudaStreamManager streams(runtime);
    if (!runtime->available()) {
        expect(!streams.available() &&
                   streams.stream(hypermoe::backend::CudaStreamRole::Compute) == nullptr,
               "stream manager has a clean unavailable state");
    }
    expect(hypermoe::backend::toString(
               hypermoe::backend::CudaStreamRole::Prefetch) == "PREFETCH",
           "stream roles have stable diagnostic names");
    expect(hypermoe::scheduler::toString(
               hypermoe::scheduler::RuntimeEventType::CudaTransferStarted) ==
               "CUDA_TRANSFER_STARTED" &&
               hypermoe::scheduler::toString(
                   hypermoe::scheduler::RuntimeEventType::CudaKernelCompleted) ==
                   "CUDA_KERNEL_COMPLETED",
           "runtime event system exposes CUDA transfer and kernel events");
}

void testCudaWhenAvailable() {
    hypermoe::backend::CudaRuntime runtime;
    if (!runtime.available()) {
        std::cout << "Phase 4 CUDA tests skipped: runtime/device unavailable\n";
        return;
    }

    const auto before = runtime.deviceInfo();
    expect(!before.name.empty() && before.totalVramBytes > 0 &&
               before.freeVramBytes > 0 && !before.computeCapability().empty(),
           "CUDA runtime reports device name, capability, and VRAM");

    auto managedRuntime = std::make_shared<hypermoe::backend::CudaRuntime>();
    hypermoe::backend::CudaStreamManager streams(managedRuntime);
    expect(streams.available() && managedRuntime->deviceInfo().streamCount == 3,
           "CUDA stream manager creates compute, transfer, and prefetch streams");
    const auto eventStart = managedRuntime->createEvent(true);
    const auto eventEnd = managedRuntime->createEvent(true);
    managedRuntime->recordEvent(
        eventStart, streams.stream(hypermoe::backend::CudaStreamRole::Compute));
    managedRuntime->recordEvent(
        eventEnd, streams.stream(hypermoe::backend::CudaStreamRole::Compute));
    managedRuntime->synchronizeEvent(eventEnd);
    expect(managedRuntime->elapsedMilliseconds(eventStart, eventEnd) >= 0.0F,
           "CUDA runtime owns, records, synchronizes, and times events");
    managedRuntime->destroyEvent(eventStart);
    managedRuntime->destroyEvent(eventEnd);

    auto backend = std::make_shared<hypermoe::backend::CudaBackend>();
    hypermoe::backend::CudaMemoryPool pool(backend, 1U << 20U);
    constexpr std::size_t bytes = 64 * 1024;
    hypermoe::PinnedBuffer host(bytes, backend);
    std::fill(host.bytes().begin(), host.bytes().end(), std::byte{0x5c});
    auto device = pool.allocate(bytes);
    backend->copyToDevice(device.data(), host.data(), bytes,
                          streams.stream(hypermoe::backend::CudaStreamRole::Transfer));
    std::vector<std::byte> output(bytes);
    backend->copyFromDevice(output.data(), device.data(), bytes,
                            streams.stream(hypermoe::backend::CudaStreamRole::Transfer));
    streams.synchronize(hypermoe::backend::CudaStreamRole::Transfer);
    expect(output == std::vector<std::byte>(bytes, std::byte{0x5c}),
           "CUDA pooled allocation preserves an asynchronous transfer round trip");
    void* pointer = device.data();
    device.reset();
    auto reused = pool.allocate(bytes);
    expect(reused.data() == pointer && pool.stats().reuseCount == 1,
           "CUDA allocation is reused after RAII release");
}

} // namespace

int main() {
    testRuntimeFallbackAndCapabilities();
    testMemoryPoolWithCpuBackend();
    testPoolBufferOutlivesPool();
    testStreamRolesAndRuntimeEvents();
    testCudaWhenAvailable();
    if (failures != 0) {
        std::cerr << failures << " Phase 4 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 4 tests passed\n";
    return EXIT_SUCCESS;
}
