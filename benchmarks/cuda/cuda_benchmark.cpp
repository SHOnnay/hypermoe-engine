#include "backend/CudaBackend.hpp"
#include "backend/cuda/CudaMemoryPool.hpp"
#include "backend/cuda/CudaRuntime.hpp"
#include "backend/cuda/CudaStreamManager.hpp"
#include "memory/PinnedBuffer.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t MiB = 1024 * 1024;

struct Options {
    std::filesystem::path report{"cuda_report.json"};
    std::size_t bytes{64 * MiB};
    std::uint32_t iterations{16};
};

struct Results {
    bool cudaCompiled{};
    bool cudaAvailable{};
    std::string skipReason;
    hypermoe::backend::DeviceInfo device;
    double directAllocationsPerSecond{};
    double pooledAllocationsPerSecond{};
    double poolReusePercentage{};
    double pinnedHostToDeviceGiBs{};
    double pageableHostToDeviceGiBs{};
    double deviceToHostGiBs{};
    double serialTwoStreamMs{};
    double concurrentTwoStreamMs{};
    double streamOverlapSpeedup{};
};

template <typename T>
T parseNumber(std::string_view value, std::string_view name) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result == 0) {
        throw std::invalid_argument("invalid value for " + std::string(name));
    }
    return result;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto separator = argument.find('=');
        if (separator == std::string_view::npos) {
            throw std::invalid_argument("expected --name=value");
        }
        const auto name = argument.substr(0, separator);
        const auto value = argument.substr(separator + 1);
        if (name == "--report") options.report = value;
        else if (name == "--bytes") options.bytes = parseNumber<std::size_t>(value, name);
        else if (name == "--iterations") {
            options.iterations = parseNumber<std::uint32_t>(value, name);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    return options;
}

double seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double bandwidth(std::uint64_t bytes, std::chrono::steady_clock::duration duration) {
    const auto elapsed = seconds(duration);
    return elapsed <= 0.0
               ? 0.0
               : static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / elapsed;
}

Results run(const Options& options) {
    Results results;
    results.cudaCompiled = hypermoe::backend::CudaRuntime::compiledWithCuda();
    auto runtime = std::make_shared<hypermoe::backend::CudaRuntime>();
    results.device = runtime->deviceInfo();
    results.cudaAvailable = runtime->available();
    if (!results.cudaAvailable) {
        results.skipReason = results.cudaCompiled
                                 ? "CUDA was compiled but no usable NVIDIA device was found"
                                 : "CUDA toolkit support was not compiled";
        return results;
    }

    hypermoe::backend::CudaStreamManager streams(runtime);
    auto backend = std::make_shared<hypermoe::backend::CudaBackend>();
    constexpr std::size_t allocationProbeBytes = 4 * MiB;
    constexpr std::uint32_t allocationIterations = 64;
    auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < allocationIterations; ++iteration) {
        void* pointer = backend->allocate(allocationProbeBytes);
        backend->free(pointer);
    }
    results.directAllocationsPerSecond =
        allocationIterations / seconds(std::chrono::steady_clock::now() - start);

    const auto doubledBytes =
        options.bytes > std::numeric_limits<std::size_t>::max() / 2
            ? std::numeric_limits<std::size_t>::max()
            : options.bytes * 2;
    hypermoe::backend::CudaMemoryPool pool(
        backend, std::max(doubledBytes, allocationProbeBytes));
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < allocationIterations; ++iteration) {
        auto buffer = pool.allocate(allocationProbeBytes);
        (void)buffer;
    }
    results.pooledAllocationsPerSecond =
        allocationIterations / seconds(std::chrono::steady_clock::now() - start);
    const auto poolStats = pool.stats();
    results.poolReusePercentage = allocationIterations == 0
                                      ? 0.0
                                      : 100.0 * static_cast<double>(poolStats.reuseCount) /
                                            allocationIterations;

    std::vector<std::byte> pageable(options.bytes, std::byte{0x37});
    hypermoe::PinnedBuffer pinnedA(options.bytes, backend);
    hypermoe::PinnedBuffer pinnedB(options.bytes, backend);
    std::fill(pinnedA.bytes().begin(), pinnedA.bytes().end(), std::byte{0x37});
    std::fill(pinnedB.bytes().begin(), pinnedB.bytes().end(), std::byte{0x19});
    auto deviceA = pool.allocate(options.bytes);
    auto deviceB = pool.allocate(options.bytes);
    const auto transferStream =
        streams.stream(hypermoe::backend::CudaStreamRole::Transfer);
    const auto prefetchStream =
        streams.stream(hypermoe::backend::CudaStreamRole::Prefetch);

    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        backend->copyToDevice(deviceA.data(), pinnedA.data(), options.bytes,
                              transferStream);
    }
    backend->synchronize(transferStream);
    results.pinnedHostToDeviceGiBs = bandwidth(
        static_cast<std::uint64_t>(options.bytes) * options.iterations,
        std::chrono::steady_clock::now() - start);

    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        backend->copyToDevice(deviceA.data(), pageable.data(), options.bytes,
                              transferStream);
    }
    backend->synchronize(transferStream);
    results.pageableHostToDeviceGiBs = bandwidth(
        static_cast<std::uint64_t>(options.bytes) * options.iterations,
        std::chrono::steady_clock::now() - start);

    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        backend->copyFromDevice(pinnedA.data(), deviceA.data(), options.bytes,
                                transferStream);
    }
    backend->synchronize(transferStream);
    results.deviceToHostGiBs = bandwidth(
        static_cast<std::uint64_t>(options.bytes) * options.iterations,
        std::chrono::steady_clock::now() - start);

    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        backend->copyToDevice(deviceA.data(), pinnedA.data(), options.bytes,
                              transferStream);
        backend->copyToDevice(deviceB.data(), pinnedB.data(), options.bytes,
                              transferStream);
    }
    backend->synchronize(transferStream);
    results.serialTwoStreamMs = milliseconds(std::chrono::steady_clock::now() - start);

    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        backend->copyToDevice(deviceA.data(), pinnedA.data(), options.bytes,
                              transferStream);
        backend->copyToDevice(deviceB.data(), pinnedB.data(), options.bytes,
                              prefetchStream);
    }
    backend->synchronize(transferStream);
    backend->synchronize(prefetchStream);
    results.concurrentTwoStreamMs = milliseconds(std::chrono::steady_clock::now() - start);
    results.streamOverlapSpeedup = results.concurrentTwoStreamMs <= 0.0
                                       ? 0.0
                                       : results.serialTwoStreamMs /
                                             results.concurrentTwoStreamMs;

    std::vector<std::byte> verification(options.bytes);
    backend->copyFromDevice(verification.data(), deviceA.data(), options.bytes,
                            transferStream);
    backend->synchronize(transferStream);
    if (verification != std::vector<std::byte>(options.bytes, std::byte{0x37})) {
        throw std::runtime_error("CUDA benchmark transfer verification failed");
    }
    results.device = runtime->deviceInfo();
    return results;
}

std::string escapeJson(std::string_view value) {
    std::string result;
    for (const char character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

std::string toJson(const Options& options, const Results& results) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"cuda_compiled\": " << (results.cudaCompiled ? "true" : "false")
           << ",\n"
           << "  \"cuda_available\": " << (results.cudaAvailable ? "true" : "false")
           << ",\n"
           << "  \"skip_reason\": \"" << escapeJson(results.skipReason) << "\",\n"
           << "  \"device\": {\n"
           << "    \"name\": \"" << escapeJson(results.device.name) << "\",\n"
           << "    \"compute_capability\": \""
           << escapeJson(results.device.computeCapability()) << "\",\n"
           << "    \"total_vram_bytes\": " << results.device.totalVramBytes << ",\n"
           << "    \"free_vram_bytes\": " << results.device.freeVramBytes << ",\n"
           << "    \"stream_count\": " << results.device.streamCount << ",\n"
           << "    \"cuda_runtime_version\": " << results.device.runtimeVersion << ",\n"
           << "    \"cuda_driver_version\": " << results.device.driverVersion << "\n"
           << "  },\n"
           << "  \"configuration\": {\n"
           << "    \"buffer_bytes\": " << options.bytes << ",\n"
           << "    \"transfer_iterations\": " << options.iterations << "\n"
           << "  },\n"
           << "  \"benchmarks\": {\n"
           << "    \"vram_direct_allocations_per_second\": "
           << results.directAllocationsPerSecond << ",\n"
           << "    \"vram_pooled_allocations_per_second\": "
           << results.pooledAllocationsPerSecond << ",\n"
           << "    \"pool_reuse_percentage\": " << results.poolReusePercentage << ",\n"
           << "    \"pinned_host_to_device_gib_s\": "
           << results.pinnedHostToDeviceGiBs << ",\n"
           << "    \"pageable_host_to_device_gib_s\": "
           << results.pageableHostToDeviceGiBs << ",\n"
           << "    \"device_to_host_gib_s\": " << results.deviceToHostGiBs << ",\n"
           << "    \"serial_copy_ms\": " << results.serialTwoStreamMs << ",\n"
           << "    \"multi_stream_copy_ms\": " << results.concurrentTwoStreamMs << ",\n"
           << "    \"multi_stream_speedup\": " << results.streamOverlapSpeedup << "\n"
           << "  }\n"
           << "}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        const auto results = run(options);
        std::ofstream report(options.report, std::ios::binary | std::ios::trunc);
        report << toJson(options, results);
        if (!report) throw std::runtime_error("failed writing CUDA report");
        std::cout << "HyperMoE CUDA benchmark\n"
                  << "  CUDA compiled:  " << (results.cudaCompiled ? "yes" : "no") << '\n'
                  << "  CUDA available: " << (results.cudaAvailable ? "yes" : "no") << '\n';
        if (results.cudaAvailable) {
            std::cout << "  GPU:            " << results.device.name << '\n'
                      << "  Pinned H2D:     " << results.pinnedHostToDeviceGiBs
                      << " GiB/s\n"
                      << "  Pageable H2D:   " << results.pageableHostToDeviceGiBs
                      << " GiB/s\n"
                      << "  D2H:            " << results.deviceToHostGiBs << " GiB/s\n"
                      << "  Stream speedup: " << results.streamOverlapSpeedup << "x\n";
        } else {
            std::cout << "  GPU metrics:    skipped (" << results.skipReason << ")\n";
        }
        std::cout << "  Report:         " << options.report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CUDA benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
