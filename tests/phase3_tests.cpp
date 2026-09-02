#include "backend/CpuBackend.hpp"
#include "backend/CudaBackend.hpp"
#include "hardware/HardwareInfo.hpp"
#include "memory/PinnedBuffer.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
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

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path() /
                ("hypermoe-phase3-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void testCpuBackend() {
    auto backend = std::make_shared<hypermoe::backend::CpuBackend>();
    constexpr std::size_t size = 16 * 1024;
    std::vector<std::byte> input(size, std::byte{0x6d});
    std::vector<std::byte> output(size);
    {
        hypermoe::backend::DeviceBuffer device(backend, size);
        backend->copyToDevice(device.data(), input.data(), size);
        backend->copyFromDevice(output.data(), device.data(), size);
        backend->synchronize();
        expect(input == output, "CPU backend round trip preserves bytes");
        expect(backend->getMemoryInfo().allocatedBytes == size,
               "CPU backend accounts for active device buffers");
    }
    const auto stats = backend->stats();
    expect(stats.allocatedBytes == 0, "device buffer automatically releases allocation");
    expect(stats.hostToDeviceBytes == size && stats.deviceToHostBytes == size,
           "CPU backend tracks both transfer directions");
}

void testPinnedFallback() {
    auto backend = std::make_shared<hypermoe::backend::CpuBackend>();
    hypermoe::PinnedBuffer buffer(4096, backend);
    expect(buffer.data() != nullptr && buffer.size() == 4096,
           "pinned buffer fallback allocates aligned host memory");
    expect(!buffer.isPinned(), "CPU fallback does not claim page-locked memory");
    buffer.bytes().front() = std::byte{0x42};
    hypermoe::PinnedBuffer moved(std::move(buffer));
    expect(moved.bytes().front() == std::byte{0x42} && buffer.data() == nullptr,
           "pinned buffer move transfers ownership");
}

void testHardwareTransferPipeline() {
    TemporaryDirectory directory;
    std::vector<std::byte> weights(32 * 1024);
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = static_cast<std::byte>(index & 0xffU);
    }
    const std::vector<hypermoe::storage::ExpertBlob> blobs{
        {0, 7, 3, weights},
    };
    hypermoe::storage::ExpertStore::create(directory.path(), blobs, "{\"phase\":3}");
    auto store = std::make_shared<hypermoe::storage::ExpertStore>(directory.path());
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
    auto backend = std::make_shared<hypermoe::backend::CpuBackend>();
    hypermoe::TransferManager transfers(loader, backend, 1);
    std::atomic_int callbacks{};

    hypermoe::TransferRequest upload;
    upload.layerId = 0;
    upload.expertId = 7;
    upload.source = hypermoe::MemoryTier::Nvme;
    upload.destination = hypermoe::MemoryTier::Vram;
    upload.priority = 5;
    upload.callback = [&callbacks](const hypermoe::TransferResult& result) {
        if (result.status == hypermoe::TransferStatus::Completed) ++callbacks;
    };
    const auto uploaded = transfers.submit(std::move(upload)).future().get();
    expect(uploaded.deviceBuffer && uploaded.deviceBuffer->size() == weights.size(),
           "NVMe to pinned staging to backend allocation completes");
    expect(uploaded.nvmeBytes == weights.size() &&
               uploaded.ramToVramBytes == weights.size(),
           "hardware transfer accounts for both hierarchy edges");

    hypermoe::TransferRequest download;
    download.layerId = 0;
    download.expertId = 7;
    download.source = hypermoe::MemoryTier::Vram;
    download.destination = hypermoe::MemoryTier::Ram;
    download.sourceDeviceBuffer = uploaded.deviceBuffer;
    const auto downloaded = transfers.submit(std::move(download)).future().get();
    expect(downloaded.buffer && *downloaded.buffer == weights,
           "backend to RAM transfer preserves expert weights");
    expect(callbacks.load() == 1, "completion callback runs exactly once");
    expect(backend->stats().hostToDeviceBytes == weights.size(),
           "pipeline uses ComputeBackend copyToDevice");
}

void testHardwareDetectionAndProfiler() {
    const auto hardware = hypermoe::hardware::detectHardware();
    expect(!hardware.operatingSystem.empty() && hardware.logicalCpuCount > 0,
           "runtime hardware detection reports CPU and OS");
    expect(hardware.ramBytes > 0 && hardware.availableStorageBytes > 0,
           "runtime hardware detection reports RAM and storage");
    const auto json = hardware.toJson();
    expect(json.find("\"cuda_available\"") != std::string::npos,
           "hardware information exports CUDA availability");

    hypermoe::Profiler profiler;
    profiler.recordCudaTransferTime(std::chrono::microseconds(10));
    profiler.recordNvmeReadTime(std::chrono::microseconds(20));
    profiler.recordRamCopyTime(std::chrono::microseconds(30));
    profiler.observeGpuMemory(4096);
    profiler.observeTransferQueueDepth(3);
    const auto report = profiler.toJson();
    expect(report.find("\"cuda_transfer_time_ms\": 0.010000") != std::string::npos,
           "profiler exports CUDA transfer timing");
    expect(report.find("\"peak_transfer_queue_depth\": 3") != std::string::npos,
           "profiler exports queue depth high-water mark");
}

void testCudaWhenAvailable() {
    const auto info = hypermoe::backend::CudaBackend::query();
    if (!info.available) {
        std::cout << "CUDA test skipped: CUDA runtime/device unavailable\n";
        return;
    }
    auto backend = std::make_shared<hypermoe::backend::CudaBackend>();
    hypermoe::PinnedBuffer pinned(4096, backend);
    std::fill(pinned.bytes().begin(), pinned.bytes().end(), std::byte{0x2a});
    hypermoe::backend::DeviceBuffer device(backend, pinned.size());
    const auto stream = backend->createStream();
    backend->copyToDevice(device.data(), pinned.data(), pinned.size(), stream);
    std::vector<std::byte> output(pinned.size());
    backend->copyFromDevice(output.data(), device.data(), output.size(), stream);
    backend->synchronize(stream);
    backend->destroyStream(stream);
    expect(output.front() == std::byte{0x2a} && pinned.isPinned(),
           "CUDA allocation, pinned memory, and async copies work");
}

} // namespace

int main() {
    testCpuBackend();
    testPinnedFallback();
    testHardwareTransferPipeline();
    testHardwareDetectionAndProfiler();
    testCudaWhenAvailable();
    if (failures != 0) {
        std::cerr << failures << " Phase 3 test assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 3 tests passed\n";
    return EXIT_SUCCESS;
}
