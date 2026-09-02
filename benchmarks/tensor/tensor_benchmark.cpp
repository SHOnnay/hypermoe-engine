#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::size_t rows{128};
    std::size_t inner{128};
    std::size_t columns{128};
    std::uint32_t iterations{5};
    std::filesystem::path report{"tensor_report.json"};
};

struct BackendResult {
    double gemmGflops{};
    double averageAllocationMicroseconds{};
    double copyGiBs{};
    double averageSynchronizeMicroseconds{};
};

struct Results {
    BackendResult cpu;
    BackendResult cuda;
    bool cudaAvailable{};
    double cudaMaximumAbsoluteError{};
    std::string cudaSkipReason;
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
        if (name == "--m") options.rows = parseNumber<std::size_t>(value, name);
        else if (name == "--k") options.inner = parseNumber<std::size_t>(value, name);
        else if (name == "--n") options.columns = parseNumber<std::size_t>(value, name);
        else if (name == "--iterations") {
            options.iterations = parseNumber<std::uint32_t>(value, name);
        } else if (name == "--report") {
            options.report = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    return options;
}

double seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

template <typename Backend>
BackendResult measureCommon(Backend& backend,
                            const hypermoe::tensor::Shape& allocationShape,
                            const hypermoe::tensor::Tensor& copySource,
                            hypermoe::tensor::Tensor& copyDestination) {
    BackendResult result;
    constexpr std::uint32_t allocationIterations = 256;
    auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < allocationIterations; ++iteration) {
        auto tensor = backend.allocateTensor(allocationShape,
                                             hypermoe::tensor::DType::FP32);
        (void)tensor;
    }
    result.averageAllocationMicroseconds =
        seconds(std::chrono::steady_clock::now() - start) * 1.0e6 /
        allocationIterations;

    constexpr std::uint32_t copyIterations = 64;
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < copyIterations; ++iteration) {
        backend.copyTensor(copySource, copyDestination);
    }
    const auto copySeconds = seconds(std::chrono::steady_clock::now() - start);
    result.copyGiBs = copySeconds <= 0.0
                          ? 0.0
                          : static_cast<double>(copySource.bytes()) * copyIterations /
                                (1024.0 * 1024.0 * 1024.0) / copySeconds;

    constexpr std::uint32_t syncIterations = 256;
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < syncIterations; ++iteration) {
        backend.synchronize();
    }
    result.averageSynchronizeMicroseconds =
        seconds(std::chrono::steady_clock::now() - start) * 1.0e6 /
        syncIterations;
    return result;
}

void initialize(hypermoe::tensor::Tensor& tensor, float phase) {
    auto* values = static_cast<float*>(tensor.data());
    for (std::size_t index = 0; index < tensor.shape().elementCount(); ++index) {
        values[index] = std::sin(static_cast<float>(index) * 0.013F + phase);
    }
}

double gemmGflops(const Options& options,
                  std::chrono::steady_clock::duration duration) {
    const auto operations = 2.0 * static_cast<double>(options.rows) *
                            static_cast<double>(options.inner) *
                            static_cast<double>(options.columns) *
                            options.iterations;
    return operations / seconds(duration) / 1.0e9;
}

Results run(const Options& options) {
    using hypermoe::tensor::DType;
    using hypermoe::tensor::Shape;
    Results results;
    hypermoe::tensor::CpuTensorBackend cpu;
    auto hostLeft = cpu.allocateTensor(Shape{options.rows, options.inner}, DType::FP32);
    auto hostRight = cpu.allocateTensor(Shape{options.inner, options.columns}, DType::FP32);
    auto hostOutput = cpu.allocateTensor(Shape{options.rows, options.columns}, DType::FP32);
    auto hostCopy = cpu.allocateTensor(hostOutput.shape(), DType::FP32);
    initialize(hostLeft, 0.1F);
    initialize(hostRight, 0.7F);
    cpu.matmul(hostLeft, hostRight, hostOutput);
    auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        cpu.matmul(hostLeft, hostRight, hostOutput);
    }
    results.cpu.gemmGflops =
        gemmGflops(options, std::chrono::steady_clock::now() - start);
    const auto commonCpu = measureCommon(cpu, Shape{256, 256}, hostOutput, hostCopy);
    results.cpu.averageAllocationMicroseconds = commonCpu.averageAllocationMicroseconds;
    results.cpu.copyGiBs = commonCpu.copyGiBs;
    results.cpu.averageSynchronizeMicroseconds =
        commonCpu.averageSynchronizeMicroseconds;

    hypermoe::tensor::CudaTensorBackend cuda;
    results.cudaAvailable = cuda.available();
    if (!results.cudaAvailable) {
        results.cudaSkipReason = "CUDA/cuBLAS runtime or NVIDIA device unavailable";
        return results;
    }

    auto deviceLeft = cuda.allocateTensor(hostLeft.shape(), DType::FP32);
    auto deviceRight = cuda.allocateTensor(hostRight.shape(), DType::FP32);
    auto deviceOutput = cuda.allocateTensor(hostOutput.shape(), DType::FP32);
    auto deviceCopy = cuda.allocateTensor(hostOutput.shape(), DType::FP32);
    cuda.copyTensor(hostLeft, deviceLeft);
    cuda.copyTensor(hostRight, deviceRight);
    cuda.matmul(deviceLeft, deviceRight, deviceOutput);
    start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        cuda.matmul(deviceLeft, deviceRight, deviceOutput);
    }
    cuda.synchronize();
    results.cuda.gemmGflops =
        gemmGflops(options, std::chrono::steady_clock::now() - start);
    const auto commonCuda =
        measureCommon(cuda, Shape{256, 256}, deviceOutput, deviceCopy);
    results.cuda.averageAllocationMicroseconds = commonCuda.averageAllocationMicroseconds;
    results.cuda.copyGiBs = commonCuda.copyGiBs;
    results.cuda.averageSynchronizeMicroseconds =
        commonCuda.averageSynchronizeMicroseconds;

    cuda.copyTensor(deviceOutput, hostCopy);
    const auto* expected = static_cast<const float*>(hostOutput.data());
    const auto* actual = static_cast<const float*>(hostCopy.data());
    for (std::size_t index = 0; index < hostOutput.shape().elementCount(); ++index) {
        results.cudaMaximumAbsoluteError =
            std::max(results.cudaMaximumAbsoluteError,
                     static_cast<double>(std::fabs(expected[index] - actual[index])));
    }
    if (results.cudaMaximumAbsoluteError > 1.0e-5) {
        throw std::runtime_error("CUDA GEMM exceeded FP32 correctness tolerance");
    }
    return results;
}

std::string escapeJson(std::string_view value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

void writeBackend(std::ostringstream& output,
                  std::string_view name,
                  const BackendResult& result,
                  bool trailingComma) {
    output << "  \"" << name << "\": {\n"
           << "    \"fp32_gemm_gflops\": " << result.gemmGflops << ",\n"
           << "    \"average_tensor_allocation_us\": "
           << result.averageAllocationMicroseconds << ",\n"
           << "    \"tensor_copy_gib_s\": " << result.copyGiBs << ",\n"
           << "    \"average_synchronize_us\": "
           << result.averageSynchronizeMicroseconds << "\n"
           << "  }" << (trailingComma ? "," : "") << "\n";
}

std::string toJson(const Options& options, const Results& results) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"configuration\": {\n"
           << "    \"m\": " << options.rows << ",\n"
           << "    \"k\": " << options.inner << ",\n"
           << "    \"n\": " << options.columns << ",\n"
           << "    \"iterations\": " << options.iterations << "\n"
           << "  },\n"
           << "  \"cuda_available\": " << (results.cudaAvailable ? "true" : "false")
           << ",\n"
           << "  \"cuda_skip_reason\": \"" << escapeJson(results.cudaSkipReason)
           << "\",\n";
    writeBackend(output, "cpu", results.cpu, true);
    writeBackend(output, "cuda", results.cuda, true);
    output << "  \"cuda_maximum_absolute_error\": "
           << results.cudaMaximumAbsoluteError << "\n"
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
        if (!report) throw std::runtime_error("failed writing tensor benchmark report");
        std::cout << std::fixed << std::setprecision(3)
                  << "HyperMoE tensor benchmark\n"
                  << "  CPU FP32 GEMM: " << results.cpu.gemmGflops << " GFLOP/s\n"
                  << "  CPU copy:      " << results.cpu.copyGiBs << " GiB/s\n"
                  << "  CUDA available:"
                  << (results.cudaAvailable ? " yes" : " no") << '\n';
        if (results.cudaAvailable) {
            std::cout << "  CUDA FP32 GEMM:" << results.cuda.gemmGflops
                      << " GFLOP/s\n"
                      << "  CUDA copy:     " << results.cuda.copyGiBs << " GiB/s\n";
        } else {
            std::cout << "  CUDA metrics:  skipped (" << results.cudaSkipReason << ")\n";
        }
        std::cout << "  Report:        " << options.report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "tensor benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
