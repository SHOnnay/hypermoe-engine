#include "backend/cuda/Int8GemmPlan.hpp"
#include "profiling/Profiler.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        using namespace hypermoe;
        using namespace tensor;
        using backend::cuda::Int8GemmMode;
        if (argc > 2) throw std::invalid_argument("usage: hypermoe_int8_gemm_benchmark [report.json]");
        CudaTensorBackend probe;
        std::ostringstream json;
        json << "{\n  \"schema\": \"hypermoe.int8-gemm-benchmark.v1\",\n"
             << "  \"cuda_available\": " << (probe.nativeKernelsAvailable() ? "true" : "false") << ",\n"
             << "  \"gpu_utilization_percent\": null,\n  \"measurements\": [";
        if (probe.nativeKernelsAvailable()) {
            CpuTensorBackend cpu;
            bool first = true;
            constexpr std::size_t repeats = 20;
            constexpr std::array<std::array<std::size_t, 3>, 3> shapes{{
                {1, 2048, 768}, {1, 768, 2048}, {1, 1031, 67}}};
            for (auto mode : {Int8GemmMode::Reference, Int8GemmMode::Cooperative}) {
                for (bool timed : {false, true}) {
                    auto profiler = timed ? std::make_shared<Profiler>() : std::shared_ptr<Profiler>{};
                    CudaTensorBackend gpu(0, profiler, mode);
                    for (const auto& dims : shapes) {
                        const auto [rows, inner, columns] = dims;
                        auto a = cpu.allocateTensor({rows, inner}, DType::FP32);
                        auto b = cpu.allocateTensor({inner, columns}, DType::INT8);
                        auto expected = cpu.allocateTensor({rows, columns}, DType::FP32);
                        auto* input = static_cast<float*>(a.data());
                        auto* weights = static_cast<std::int8_t*>(b.data());
                        for (std::size_t i = 0; i < rows * inner; ++i) input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 29.0F;
                        for (std::size_t i = 0; i < inner * columns; ++i) weights[i] = static_cast<std::int8_t>(static_cast<int>((i * 13) % 255) - 127);
                        const quantization::QuantizationParameters parameters{0.003F, -3};
                        cpu.matmulInt8Weights(a, b, parameters, expected);
                        auto ga = gpu.allocateTensor(a.shape(), a.dtype());
                        auto gb = gpu.allocateTensor(b.shape(), b.dtype());
                        auto gc = gpu.allocateTensor(expected.shape(), expected.dtype());
                        gpu.copyTensor(a, ga); gpu.copyTensor(b, gb);
                        for (int warmup = 0; warmup < 3; ++warmup) {
                            gpu.matmulInt8Expert(ga, gb, parameters, gc); gpu.synchronizeExecution();
                        }
                        const auto before = gpu.backendStats();
                        const auto gpuBefore = profiler ? profiler->snapshot().cudaExpertGemmTime : std::chrono::nanoseconds{};
                        const auto start = std::chrono::steady_clock::now();
                        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                            gpu.matmulInt8Expert(ga, gb, parameters, gc);
                            // Identical explicit latency boundary for both implementations.
                            gpu.synchronizeExecution();
                        }
                        const auto elapsed = std::chrono::steady_clock::now() - start;
                        const auto after = gpu.backendStats();
                        const auto gpuTime = profiler ? profiler->snapshot().cudaExpertGemmTime - gpuBefore : std::chrono::nanoseconds{};
                        auto actual = cpu.allocateTensor(expected.shape(), expected.dtype());
                        gpu.copyTensor(gc, actual);
                        const auto* reference = static_cast<const float*>(expected.data());
                        const auto* current = static_cast<const float*>(actual.data());
                        double maximumError{};
                        for (std::size_t i = 0; i < rows * columns; ++i) {
                            maximumError = std::max(maximumError, static_cast<double>(std::abs(current[i] - reference[i])));
                            if (!std::isfinite(current[i]) || std::abs(current[i] - reference[i]) > 1.0e-5F + 1.0e-5F * std::abs(reference[i])) {
                                throw std::runtime_error("INT8 benchmark CPU/CUDA correctness failure");
                            }
                        }
                        const auto plan = backend::cuda::planInt8Gemm(rows, inner, columns, mode);
                        json << (first ? "\n" : ",\n") << "    {\"mode\": \"" << backend::cuda::toString(mode)
                             << "\", \"profiling\": " << (timed ? "true" : "false")
                             << ", \"rows\": " << rows << ", \"inner\": " << inner << ", \"columns\": " << columns
                             << ", \"blocks\": " << plan.blocks << ", \"partitions\": " << plan.partitions
                             << ", \"scratch_bytes\": " << plan.scratchElements * sizeof(float)
                             << ", \"repeats\": " << repeats
                             << ", \"mean_wall_ms\": " << std::chrono::duration<double, std::milli>(elapsed).count() / static_cast<double>(repeats)
                             << ", \"mean_cuda_gemm_ms\": " << (timed ? std::to_string(std::chrono::duration<double, std::milli>(gpuTime).count() / static_cast<double>(repeats)) : "null")
                             << ", \"synchronization_count\": " << after.synchronizationCount - before.synchronizationCount
                             << ", \"synchronization_ms\": " << std::chrono::duration<double, std::milli>(after.synchronizationTime - before.synchronizationTime).count()
                             << ", \"transfer_ms\": " << std::chrono::duration<double, std::milli>(after.transferTime - before.transferTime).count()
                             << ", \"maximum_abs_error\": " << maximumError << "}";
                        first = false;
                    }
                }
            }
            json << '\n';
        }
        json << "  ]\n}\n";
        if (argc == 2) {
            std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
            output << json.str();
            if (!output) throw std::runtime_error("cannot write INT8 GEMM benchmark report");
        }
        std::cout << json.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "INT8 GEMM benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
