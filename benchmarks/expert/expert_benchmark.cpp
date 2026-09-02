#include "backend/CpuBackend.hpp"
#include "backend/CudaBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/cache_policy.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::size_t batch{1};
    std::size_t modelWidth{128};
    std::size_t intermediateWidth{256};
    std::uint32_t iterations{5};
    std::filesystem::path report{"expert_report.json"};
};

struct Measurements {
    bool available{};
    double expertLoadMs{};
    double tensorPreparationMs{};
    double gemmMs{};
    double activationMs{};
    double totalExecutionMs{};
    std::string skipReason;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-expert-benchmark-" +
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
        if (name == "--batch") options.batch = parseNumber<std::size_t>(value, name);
        else if (name == "--model-width") {
            options.modelWidth = parseNumber<std::size_t>(value, name);
        } else if (name == "--intermediate-width") {
            options.intermediateWidth = parseNumber<std::size_t>(value, name);
        } else if (name == "--iterations") {
            options.iterations = parseNumber<std::uint32_t>(value, name);
        } else if (name == "--report") {
            options.report = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    return options;
}

double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

std::size_t checkedMultiply(std::size_t left,
                            std::size_t right,
                            std::string_view description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(description) + " size overflow");
    }
    return left * right;
}

std::vector<std::byte> makeExpertWeights(const Options& options) {
    const auto gateElements = checkedMultiply(
        options.modelWidth, options.intermediateWidth, "expert projection");
    const auto upElements = gateElements;
    const auto downElements = gateElements;
    if (gateElements > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::overflow_error("expert parameter count overflow");
    }
    std::vector<float> values(gateElements + upElements + downElements);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = std::sin(static_cast<float>(index) * 0.001F) * 0.05F;
    }
    std::vector<std::byte> bytes(
        checkedMultiply(values.size(), sizeof(float), "expert weight bytes"));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

Measurements runBackend(
    const Options& options,
    const std::filesystem::path& modelDirectory,
    std::shared_ptr<hypermoe::backend::ComputeBackend> compute,
    std::shared_ptr<hypermoe::tensor::TensorBackend> tensors) {
    using namespace hypermoe::tensor;
    Measurements result;
    result.available = true;
    auto profiler = std::make_shared<hypermoe::Profiler>();
    auto store = std::make_shared<hypermoe::storage::ExpertStore>(modelDirectory);
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
    auto transfers = std::make_shared<hypermoe::TransferManager>(loader, compute, 1);
    const auto record = store->index().find(0, 1);
    if (!record || record->size > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::overflow_error("expert record is too large for benchmark memory");
    }
    const auto weightBytes = static_cast<std::size_t>(record->size);
    hypermoe::MemoryManager memory(weightBytes * 2, weightBytes * 2);
    hypermoe::ExpertManager experts(
        memory, std::make_unique<hypermoe::LruCachePolicy>(), transfers);
    experts.registerExpert({1, 0, static_cast<std::size_t>(weightBytes),
                            hypermoe::QuantizationType::Fp32,
                            hypermoe::MemoryTier::Nvme});

    auto start = std::chrono::steady_clock::now();
    (void)experts.requestExpert(1);
    result.expertLoadMs = milliseconds(std::chrono::steady_clock::now() - start);

    const auto matrixElements = checkedMultiply(
        options.modelWidth, options.intermediateWidth, "projection matrix");
    const auto matrixBytes = checkedMultiply(
        matrixElements, sizeof(float), "projection matrix bytes");
    start = std::chrono::steady_clock::now();
    const auto raw = experts.residentDeviceTensorView(
        1, Shape{weightBytes}, DType::INT8);
    const hypermoe::ExpertMlpWeights weights{
        raw.sliceBytes(0, Shape{options.modelWidth, options.intermediateWidth},
                       DType::FP32),
        raw.sliceBytes(matrixBytes,
                       Shape{options.modelWidth, options.intermediateWidth},
                       DType::FP32),
        raw.sliceBytes(2 * matrixBytes,
                       Shape{options.intermediateWidth, options.modelWidth},
                       DType::FP32),
    };
    result.tensorPreparationMs =
        milliseconds(std::chrono::steady_clock::now() - start);

    auto input = tensors->allocateTensor({options.batch, options.modelWidth}, DType::FP32);
    auto output = tensors->allocateTensor({options.batch, options.modelWidth}, DType::FP32);
    hypermoe::tensor::CpuTensorBackend cpu;
    auto hostInput = cpu.allocateTensor(input.shape(), DType::FP32);
    auto* inputValues = static_cast<float*>(hostInput.data());
    for (std::size_t index = 0; index < hostInput.shape().elementCount(); ++index) {
        inputValues[index] = std::cos(static_cast<float>(index) * 0.01F);
    }
    tensors->copyTensor(hostInput, input);

    hypermoe::ExpertMlpExecutor executor(
        tensors, hypermoe::tensor::activation::ActivationType::SiLU, profiler);
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        executor.execute(input, weights, output);
    }
    const auto metrics = profiler->snapshot();
    const auto divisor = static_cast<double>(options.iterations);
    result.gemmMs = milliseconds(metrics.projectionTime) / divisor;
    result.activationMs = milliseconds(metrics.activationTime) / divisor;
    result.totalExecutionMs = milliseconds(metrics.expertExecutionTime) / divisor;
    return result;
}

std::string escapeJson(std::string_view value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

void writeMeasurements(std::ostringstream& output,
                       std::string_view name,
                       const Measurements& result,
                       bool trailingComma) {
    output << "  \"" << name << "\": {\n"
           << "    \"available\": " << (result.available ? "true" : "false") << ",\n"
           << "    \"skip_reason\": \"" << escapeJson(result.skipReason) << "\",\n"
           << "    \"expert_load_ms\": " << result.expertLoadMs << ",\n"
           << "    \"tensor_preparation_ms\": " << result.tensorPreparationMs << ",\n"
           << "    \"gemm_ms\": " << result.gemmMs << ",\n"
           << "    \"activation_ms\": " << result.activationMs << ",\n"
           << "    \"total_expert_execution_ms\": " << result.totalExecutionMs << "\n"
           << "  }" << (trailingComma ? "," : "") << '\n';
}

std::string toJson(const Options& options,
                   const Measurements& cpu,
                   const Measurements& cuda) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n  \"configuration\": {\n"
           << "    \"batch\": " << options.batch << ",\n"
           << "    \"model_width\": " << options.modelWidth << ",\n"
           << "    \"intermediate_width\": " << options.intermediateWidth << ",\n"
           << "    \"iterations\": " << options.iterations << "\n  },\n";
    writeMeasurements(output, "cpu", cpu, true);
    writeMeasurements(output, "cuda", cuda, false);
    output << "}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        TemporaryDirectory directory;
        const auto weights = makeExpertWeights(options);
        const std::vector<hypermoe::storage::ExpertBlob> blobs{
            {0, 1, static_cast<std::uint32_t>(hypermoe::QuantizationType::Fp32),
             weights},
        };
        hypermoe::storage::ExpertStore::create(
            directory.path(), blobs, "{\"benchmark\":\"expert\"}");

        auto cpuCompute = std::make_shared<hypermoe::backend::CpuBackend>();
        auto cpuTensors =
            std::make_shared<hypermoe::tensor::CpuTensorBackend>();
        const auto cpu = runBackend(options, directory.path(), cpuCompute, cpuTensors);

        Measurements cuda;
        auto cudaTensors =
            std::make_shared<hypermoe::tensor::CudaTensorBackend>();
        if (cudaTensors->available()) {
            auto cudaCompute = std::make_shared<hypermoe::backend::CudaBackend>();
            cuda = runBackend(options, directory.path(), cudaCompute, cudaTensors);
        } else {
            cuda.skipReason = "CUDA/cuBLAS runtime or NVIDIA device unavailable";
        }

        std::ofstream report(options.report, std::ios::binary | std::ios::trunc);
        report << toJson(options, cpu, cuda);
        if (!report) throw std::runtime_error("failed writing expert benchmark report");
        std::cout << std::fixed << std::setprecision(3)
                  << "HyperMoE expert benchmark\n"
                  << "  CPU expert load: " << cpu.expertLoadMs << " ms\n"
                  << "  CPU preparation: " << cpu.tensorPreparationMs << " ms\n"
                  << "  CPU GEMM:        " << cpu.gemmMs << " ms/expert\n"
                  << "  CPU activation:  " << cpu.activationMs << " ms/expert\n"
                  << "  CPU total:       " << cpu.totalExecutionMs << " ms/expert\n"
                  << "  CUDA available:  " << (cuda.available ? "yes" : "no") << '\n';
        if (cuda.available) {
            std::cout << "  CUDA total:      " << cuda.totalExecutionMs << " ms/expert\n";
        } else {
            std::cout << "  CUDA metrics:    skipped (" << cuda.skipReason << ")\n";
        }
        std::cout << "  Report:          " << options.report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "expert benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
