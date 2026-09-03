#include "backend/CudaBackend.hpp"
#include "backend/cuda/CudaRuntimeValidator.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert.hpp"
#include "memory/PinnedBuffer.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "scheduler/RuntimeEvent.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t MiB = 1024U * 1024U;

struct Results {
    hypermoe::backend::CudaRuntimeValidationReport hardware;
    bool skipped{};
    std::string skipReason;
    double allocationLatencyUs{};
    double hostToDeviceGiBs{};
    double deviceToHostGiBs{};
    double gemmGflops{};
    double cpuExpertMs{};
    double cudaExpertMs{};
    double pipelineWallMs{};
    double transferOverlapPercent{};
    double averageQueueWaitMs{};
    double averageTransferLatencyMs{};
    std::uint64_t vramAllocatedBytes{};
    std::uint64_t peakVramAllocatedBytes{};
    std::uint64_t residentExperts{};
    std::uint64_t evictionEvents{};
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-cuda-runtime-benchmark-" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double gibPerSecond(std::uint64_t bytes,
                    std::chrono::steady_clock::duration duration) {
    const auto seconds = std::chrono::duration<double>(duration).count();
    return seconds <= 0.0
               ? 0.0
               : static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) /
                     seconds;
}

void fill(hypermoe::tensor::Tensor& tensor, float seed) {
    auto* values = static_cast<float*>(tensor.data());
    for (std::size_t index = 0; index < tensor.shape().elementCount(); ++index) {
        values[index] = seed + static_cast<float>(index % 17U) * 0.001F;
    }
}

Results run() {
    using namespace hypermoe;
    Results results;
    results.hardware = backend::CudaRuntimeValidator::validate();
    tensor::CpuTensorBackend cpu;
    constexpr std::size_t hidden = 128;
    constexpr std::size_t intermediate = 256;
    constexpr std::uint32_t expertIterations = 10;
    auto hostInput = cpu.allocateTensor({1, hidden}, tensor::DType::FP32);
    auto hostGate = cpu.allocateTensor({hidden, intermediate}, tensor::DType::FP32);
    auto hostUp = cpu.allocateTensor({hidden, intermediate}, tensor::DType::FP32);
    auto hostDown = cpu.allocateTensor({intermediate, hidden}, tensor::DType::FP32);
    auto hostOutput = cpu.allocateTensor({1, hidden}, tensor::DType::FP32);
    fill(hostInput, 0.01F);
    fill(hostGate, 0.002F);
    fill(hostUp, 0.003F);
    fill(hostDown, 0.004F);
    auto cpuBackend = std::make_shared<tensor::CpuTensorBackend>();
    ExpertMlpExecutor cpuExecutor(cpuBackend);
    auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < expertIterations; ++iteration) {
        cpuExecutor.execute(hostInput, {hostGate, hostUp, hostDown}, hostOutput);
    }
    results.cpuExpertMs = milliseconds(std::chrono::steady_clock::now() - start) /
                          static_cast<double>(expertIterations);
    if (!results.hardware.passed()) {
        results.skipped = true;
        results.skipReason = results.hardware.message;
        return results;
    }

    auto rawCuda = std::make_shared<backend::CudaBackend>();
    constexpr std::size_t allocationBytes = 4U * MiB;
    constexpr std::uint32_t allocationIterations = 32;
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < allocationIterations; ++iteration) {
        void* pointer = rawCuda->allocate(allocationBytes);
        rawCuda->free(pointer);
    }
    results.allocationLatencyUs =
        milliseconds(std::chrono::steady_clock::now() - start) * 1000.0 /
        static_cast<double>(allocationIterations);

    constexpr std::size_t transferBytes = 32U * MiB;
    constexpr std::uint32_t transferIterations = 8;
    PinnedBuffer pinned(transferBytes, rawCuda);
    std::fill(pinned.bytes().begin(), pinned.bytes().end(), std::byte{0x53});
    backend::DeviceBuffer device(rawCuda, transferBytes);
    const auto stream = rawCuda->createStream();
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < transferIterations; ++iteration) {
        rawCuda->copyToDevice(device.data(), pinned.data(), transferBytes, stream);
    }
    rawCuda->synchronize(stream);
    results.hostToDeviceGiBs = gibPerSecond(
        static_cast<std::uint64_t>(transferBytes) * transferIterations,
        std::chrono::steady_clock::now() - start);
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < transferIterations; ++iteration) {
        rawCuda->copyFromDevice(pinned.data(), device.data(), transferBytes, stream);
    }
    rawCuda->synchronize(stream);
    results.deviceToHostGiBs = gibPerSecond(
        static_cast<std::uint64_t>(transferBytes) * transferIterations,
        std::chrono::steady_clock::now() - start);
    rawCuda->destroyStream(stream);

    constexpr std::size_t matrixSize = 256;
    constexpr std::uint32_t gemmIterations = 20;
    auto cuda = std::make_shared<tensor::CudaTensorBackend>();
    auto hostLeft = cpu.allocateTensor({matrixSize, matrixSize}, tensor::DType::FP32);
    auto hostRight = cpu.allocateTensor({matrixSize, matrixSize}, tensor::DType::FP32);
    fill(hostLeft, 0.01F);
    fill(hostRight, 0.02F);
    auto left = cuda->allocateTensor(hostLeft.shape(), tensor::DType::FP32);
    auto right = cuda->allocateTensor(hostRight.shape(), tensor::DType::FP32);
    auto product = cuda->allocateTensor(hostLeft.shape(), tensor::DType::FP32);
    cuda->copyTensor(hostLeft, left);
    cuda->copyTensor(hostRight, right);
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < gemmIterations; ++iteration) {
        cuda->matmul(left, right, product);
    }
    cuda->synchronize();
    const auto gemmSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto operations = 2.0 * static_cast<double>(matrixSize) *
                            static_cast<double>(matrixSize) *
                            static_cast<double>(matrixSize) *
                            static_cast<double>(gemmIterations);
    results.gemmGflops = operations / gemmSeconds / 1.0e9;

    auto cudaInput = cuda->allocateTensor(hostInput.shape(), tensor::DType::FP32);
    auto cudaGate = cuda->allocateTensor(hostGate.shape(), tensor::DType::FP32);
    auto cudaUp = cuda->allocateTensor(hostUp.shape(), tensor::DType::FP32);
    auto cudaDown = cuda->allocateTensor(hostDown.shape(), tensor::DType::FP32);
    auto cudaOutput = cuda->allocateTensor(hostOutput.shape(), tensor::DType::FP32);
    cuda->copyTensor(hostInput, cudaInput);
    cuda->copyTensor(hostGate, cudaGate);
    cuda->copyTensor(hostUp, cudaUp);
    cuda->copyTensor(hostDown, cudaDown);
    ExpertMlpExecutor cudaExecutor(cuda);
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < expertIterations; ++iteration) {
        cudaExecutor.execute(cudaInput, {cudaGate, cudaUp, cudaDown}, cudaOutput);
    }
    results.cudaExpertMs = milliseconds(std::chrono::steady_clock::now() - start) /
                           static_cast<double>(expertIterations);

    TemporaryDirectory temporary;
    std::vector<std::byte> expertA(8U * MiB, std::byte{0x19});
    std::vector<std::byte> expertB(8U * MiB, std::byte{0x37});
    const std::vector<storage::ExpertBlob> blobs{
        {0, 0, static_cast<std::uint32_t>(QuantizationType::Fp32), expertA},
        {0, 1, static_cast<std::uint32_t>(QuantizationType::Fp32), expertB}};
    storage::ExpertStore::create(
        temporary.path(), blobs,
        "{\"benchmark\":\"phase11_cuda_runtime\"}");
    auto store = std::make_shared<storage::ExpertStore>(temporary.path());
    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto transfers = std::make_shared<TransferManager>(loader, rawCuda, 2);
    auto profiler = std::make_shared<Profiler>();
    scheduler::Scheduler scheduler(transfers, profiler, 2);
    scheduler.registerExpert(0, 0);
    scheduler.registerExpert(0, 1);
    std::atomic_uint64_t evictionEvents{};
    const auto subscription = scheduler.events().subscribe(
        [&](const scheduler::RuntimeEvent& event) {
            if (event.type == scheduler::RuntimeEventType::CacheEvicted) {
                evictionEvents.fetch_add(1, std::memory_order_relaxed);
            }
        });
    scheduler::ScheduleRequest active;
    active.layerId = 0;
    active.expertId = 0;
    active.destination = MemoryTier::Vram;
    active.priority = scheduler::TransferPriority::ActiveInference;
    auto prefetch = active;
    prefetch.expertId = 1;
    prefetch.priority = scheduler::TransferPriority::PredictedNextLayer;
    start = std::chrono::steady_clock::now();
    auto activeHandle = scheduler.schedule(active);
    auto prefetchHandle = scheduler.schedule(prefetch);
    const auto activeResult = activeHandle.future().get();
    const auto prefetchResult = prefetchHandle.future().get();
    const auto pipelineDuration = std::chrono::steady_clock::now() - start;
    if (!activeResult.success || !prefetchResult.success) {
        throw std::runtime_error("CUDA transfer pipeline benchmark failed");
    }
    results.pipelineWallMs = milliseconds(pipelineDuration);
    const auto transferSum = activeResult.transfer.elapsed +
                             prefetchResult.transfer.elapsed;
    const auto transferSumMs = milliseconds(transferSum);
    results.transferOverlapPercent = transferSumMs <= 0.0
                                         ? 0.0
                                         : 100.0 * std::max(
                                               0.0, 1.0 - results.pipelineWallMs /
                                                              transferSumMs);
    results.averageTransferLatencyMs = transferSumMs / 2.0;
    results.averageQueueWaitMs = profiler->snapshot().averageQueueWaitMs();
    results.residentExperts = 2;

    scheduler::ScheduleRequest eviction;
    eviction.layerId = 0;
    eviction.expertId = 0;
    eviction.source = MemoryTier::Vram;
    eviction.destination = MemoryTier::Ram;
    eviction.priority = scheduler::TransferPriority::BackgroundMaintenance;
    eviction.deviceBuffer = activeResult.transfer.deviceBuffer;
    eviction.eviction = true;
    const auto evicted = scheduler.schedule(std::move(eviction)).future().get();
    if (!evicted.success) throw std::runtime_error("CUDA eviction benchmark failed");
    results.residentExperts = 1;
    results.evictionEvents = evictionEvents.load(std::memory_order_relaxed);
    (void)scheduler.events().unsubscribe(subscription);
    const auto backendStats = rawCuda->stats();
    results.vramAllocatedBytes = backendStats.allocatedBytes;
    results.peakVramAllocatedBytes = backendStats.peakAllocatedBytes;
    return results;
}

std::string jsonEscape(const std::string& value) {
    std::string result;
    for (const char character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

std::string toJson(const Results& results) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"schema\": \"hypermoe.cuda-runtime-benchmark.v1\",\n"
           << "  \"benchmark_kind\": \"hardware\",\n"
           << "  \"skipped\": " << (results.skipped ? "true" : "false") << ",\n"
           << "  \"skip_reason\": \"" << jsonEscape(results.skipReason) << "\",\n"
           << "  \"hardware_validation\": " << results.hardware.toJson() << ",\n"
           << "  \"allocation_latency_us\": " << results.allocationLatencyUs << ",\n"
           << "  \"h2d_gib_s\": " << results.hostToDeviceGiBs << ",\n"
           << "  \"d2h_gib_s\": " << results.deviceToHostGiBs << ",\n"
           << "  \"gemm_gflops\": " << results.gemmGflops << ",\n"
           << "  \"cpu_expert_ms\": " << results.cpuExpertMs << ",\n"
           << "  \"cuda_expert_ms\": " << results.cudaExpertMs << ",\n"
           << "  \"pipeline_wall_ms\": " << results.pipelineWallMs << ",\n"
           << "  \"transfer_overlap_percent\": "
           << results.transferOverlapPercent << ",\n"
           << "  \"average_queue_wait_ms\": " << results.averageQueueWaitMs << ",\n"
           << "  \"average_transfer_latency_ms\": "
           << results.averageTransferLatencyMs << ",\n"
           << "  \"vram_allocated_bytes\": " << results.vramAllocatedBytes << ",\n"
           << "  \"peak_vram_allocated_bytes\": "
           << results.peakVramAllocatedBytes << ",\n"
           << "  \"resident_experts\": " << results.residentExperts << ",\n"
           << "  \"eviction_events\": " << results.evictionEvents << "\n"
           << "}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path reportPath =
            argc > 1 ? argv[1] : "cuda_runtime_report.json";
        const auto results = run();
        std::ofstream output(reportPath, std::ios::binary | std::ios::trunc);
        output << toJson(results);
        if (!output) throw std::runtime_error("failed writing CUDA runtime report");
        std::cout << "HyperMoE Phase 11 CUDA runtime benchmark\n"
                  << "  Status: "
                  << (results.skipped ? "skipped" : "completed") << '\n';
        if (results.skipped) {
            std::cout << "  Reason: " << results.skipReason << '\n';
        } else {
            std::cout << "  GPU: " << results.hardware.device.name << '\n'
                      << "  H2D: " << results.hostToDeviceGiBs << " GiB/s\n"
                      << "  D2H: " << results.deviceToHostGiBs << " GiB/s\n"
                      << "  GEMM: " << results.gemmGflops << " GFLOP/s\n"
                      << "  CUDA expert: " << results.cudaExpertMs << " ms\n"
                      << "  Transfer overlap: " << results.transferOverlapPercent
                      << "%\n";
        }
        std::cout << "  Report: " << reportPath << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CUDA runtime benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
