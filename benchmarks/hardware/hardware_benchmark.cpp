#include "backend/CpuBackend.hpp"
#include "backend/CudaBackend.hpp"
#include "hardware/HardwareInfo.hpp"
#include "memory/PinnedBuffer.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t MiB = 1024 * 1024;
constexpr std::size_t kBufferSize = 32 * MiB;
volatile std::uint64_t benchmarkSink = 0;

#if defined(_MSC_VER)
#define HYPERMOE_NOINLINE __declspec(noinline)
#else
#define HYPERMOE_NOINLINE __attribute__((noinline))
#endif

HYPERMOE_NOINLINE void measuredCopy(void* destination,
                                    const void* source,
                                    std::size_t size) {
    std::memcpy(destination, source, size);
}

HYPERMOE_NOINLINE void measuredFill(std::byte* destination,
                                    std::size_t size,
                                    std::byte value) {
    std::fill(destination, destination + size, value);
}

struct Results {
    double cpuCopyGiBs{};
    double ramWriteGiBs{};
    double ramReadGiBs{};
    double nvmeSequentialGiBs{};
    double nvmeRandomMiBs{};
    double pinnedHostCopyGiBs{};
    double gpuHostToDeviceGiBs{};
    double gpuDeviceToHostGiBs{};
    bool pinnedMemoryHardwareBacked{};
    bool gpuBenchmarked{};
};

double seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

double gibPerSecond(std::uint64_t bytes, std::chrono::steady_clock::duration duration) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / seconds(duration);
}

class TemporaryFile {
public:
    TemporaryFile()
        : path_(std::filesystem::temp_directory_path() /
                ("hypermoe-hardware-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".bin")) {}
    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

Results runBenchmarks(const hypermoe::hardware::HardwareInfo& hardware) {
    Results results;
    std::vector<std::byte> source(kBufferSize, std::byte{0x5a});
    std::vector<std::byte> destination(kBufferSize);
    constexpr std::uint64_t iterations = 16;

    auto start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
        measuredCopy(destination.data(), source.data(), source.size());
    }
    results.cpuCopyGiBs = gibPerSecond(iterations * source.size(),
                                       std::chrono::steady_clock::now() - start);

    start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
        measuredFill(destination.data(), destination.size(), static_cast<std::byte>(index));
    }
    results.ramWriteGiBs = gibPerSecond(iterations * destination.size(),
                                        std::chrono::steady_clock::now() - start);

    std::uint64_t checksum = 0;
    start = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t index = 0; index < source.size(); index += 64) {
            checksum += std::to_integer<unsigned char>(source[index]);
        }
    }
    benchmarkSink = checksum;
    results.ramReadGiBs = gibPerSecond(iterations * source.size(),
                                       std::chrono::steady_clock::now() - start);
    if (checksum == 0) throw std::runtime_error("RAM benchmark checksum failed");

    TemporaryFile temporary;
    {
        std::ofstream output(temporary.path(), std::ios::binary | std::ios::trunc);
        for (std::size_t written = 0; written < kBufferSize; written += source.size()) {
            output.write(reinterpret_cast<const char*>(source.data()),
                         static_cast<std::streamsize>(source.size()));
        }
        if (!output) throw std::runtime_error("failed creating NVMe benchmark file");
    }
    {
        std::ifstream input(temporary.path(), std::ios::binary);
        start = std::chrono::steady_clock::now();
        input.read(reinterpret_cast<char*>(destination.data()),
                   static_cast<std::streamsize>(destination.size()));
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (!input) throw std::runtime_error("sequential benchmark read failed");
        results.nvmeSequentialGiBs = gibPerSecond(destination.size(), elapsed);
    }
    {
        constexpr std::size_t blockSize = 4096;
        constexpr std::uint64_t reads = 2048;
        std::vector<std::byte> block(blockSize);
        std::mt19937 random(20260902);
        std::uniform_int_distribution<std::uint64_t> offset(
            0, static_cast<std::uint64_t>(kBufferSize / blockSize - 1));
        std::ifstream input(temporary.path(), std::ios::binary);
        start = std::chrono::steady_clock::now();
        for (std::uint64_t read = 0; read < reads; ++read) {
            input.seekg(static_cast<std::streamoff>(offset(random) * blockSize));
            input.read(reinterpret_cast<char*>(block.data()),
                       static_cast<std::streamsize>(block.size()));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (!input) throw std::runtime_error("random benchmark read failed");
        results.nvmeRandomMiBs =
            static_cast<double>(reads * blockSize) / (1024.0 * 1024.0) / seconds(elapsed);
    }

    std::shared_ptr<hypermoe::backend::ComputeBackend> backend;
    if (hardware.cudaAvailable) backend = std::make_shared<hypermoe::backend::CudaBackend>();
    else backend = std::make_shared<hypermoe::backend::CpuBackend>();
    hypermoe::PinnedBuffer pinned(kBufferSize, backend);
    results.pinnedMemoryHardwareBacked = pinned.isPinned();
    start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
        measuredCopy(pinned.data(), source.data(), source.size());
    }
    results.pinnedHostCopyGiBs = gibPerSecond(iterations * source.size(),
                                              std::chrono::steady_clock::now() - start);

    if (hardware.cudaAvailable) {
        hypermoe::backend::DeviceBuffer device(backend, kBufferSize);
        const auto stream = backend->createStream();
        start = std::chrono::steady_clock::now();
        for (std::uint64_t index = 0; index < iterations; ++index) {
            backend->copyToDevice(device.data(), pinned.data(), pinned.size(), stream);
        }
        backend->synchronize(stream);
        results.gpuHostToDeviceGiBs =
            gibPerSecond(iterations * pinned.size(), std::chrono::steady_clock::now() - start);

        start = std::chrono::steady_clock::now();
        for (std::uint64_t index = 0; index < iterations; ++index) {
            backend->copyFromDevice(pinned.data(), device.data(), pinned.size(), stream);
        }
        backend->synchronize(stream);
        results.gpuDeviceToHostGiBs =
            gibPerSecond(iterations * pinned.size(), std::chrono::steady_clock::now() - start);
        backend->destroyStream(stream);
        results.gpuBenchmarked = true;
    }
    return results;
}

std::string reportJson(const hypermoe::hardware::HardwareInfo& hardware,
                       const Results& results) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n  \"hardware\": " << hardware.toJson() << ",\n"
           << "  \"benchmarks\": {\n"
           << "    \"cpu_memory_copy_gib_s\": " << results.cpuCopyGiBs << ",\n"
           << "    \"ram_write_gib_s\": " << results.ramWriteGiBs << ",\n"
           << "    \"ram_sampled_read_gib_s\": " << results.ramReadGiBs << ",\n"
           << "    \"nvme_buffered_sequential_read_gib_s\": "
           << results.nvmeSequentialGiBs << ",\n"
           << "    \"nvme_buffered_random_read_mib_s\": "
           << results.nvmeRandomMiBs << ",\n"
           << "    \"pinned_host_copy_gib_s\": " << results.pinnedHostCopyGiBs << ",\n"
           << "    \"pinned_memory_hardware_backed\": "
           << (results.pinnedMemoryHardwareBacked ? "true" : "false") << ",\n"
           << "    \"gpu_benchmarked\": "
           << (results.gpuBenchmarked ? "true" : "false") << ",\n"
           << "    \"gpu_host_to_device_gib_s\": "
           << results.gpuHostToDeviceGiBs << ",\n"
           << "    \"gpu_device_to_host_gib_s\": "
           << results.gpuDeviceToHostGiBs << "\n"
           << "  },\n"
           << "  \"notes\": \"NVMe results use buffered filesystem I/O and may include OS cache.\"\n"
           << "}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path report = "hardware_report.json";
        if (argc == 2) {
            constexpr std::string_view prefix = "--report=";
            const std::string_view argument(argv[1]);
            if (!argument.starts_with(prefix)) throw std::invalid_argument("expected --report=PATH");
            report = argument.substr(prefix.size());
        } else if (argc > 2) {
            throw std::invalid_argument("only --report=PATH is supported");
        }
        const auto hardware = hypermoe::hardware::detectHardware();
        const auto results = runBenchmarks(hardware);
        std::ofstream output(report, std::ios::binary | std::ios::trunc);
        output << reportJson(hardware, results);
        if (!output) throw std::runtime_error("failed writing hardware report");
        std::cout << "Hardware benchmark complete\n"
                  << "  CPU copy:       " << results.cpuCopyGiBs << " GiB/s\n"
                  << "  RAM write:      " << results.ramWriteGiBs << " GiB/s\n"
                  << "  NVMe sequential:" << results.nvmeSequentialGiBs << " GiB/s (buffered)\n"
                  << "  NVMe random:    " << results.nvmeRandomMiBs << " MiB/s (buffered)\n"
                  << "  Pinned copy:    " << results.pinnedHostCopyGiBs << " GiB/s\n"
                  << "  CUDA available: " << (hardware.cudaAvailable ? "yes" : "no") << '\n'
                  << "  Report:         " << report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "hardware benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
