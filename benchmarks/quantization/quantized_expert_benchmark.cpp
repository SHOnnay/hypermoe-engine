#include "experts/ExpertExecutor.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/quantization/Quantization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct CacheResult {
    std::uint64_t requests{};
    std::uint64_t hits{};
    std::uint64_t transferredBytes{};
    std::size_t capacity{};
};

CacheResult simulateCache(std::size_t expertBytes, std::size_t budgetBytes) {
    constexpr std::uint32_t expertCount = 64;
    constexpr std::uint64_t requests = 2000;
    CacheResult result;
    result.requests = requests;
    result.capacity = budgetBytes / expertBytes;
    std::list<std::uint32_t> resident;
    std::unordered_map<std::uint32_t, std::list<std::uint32_t>::iterator> positions;
    for (std::uint64_t token = 0; token < requests; ++token) {
        const auto phase = static_cast<std::uint32_t>((token / 40U) % 8U);
        const auto expert = static_cast<std::uint32_t>(
            (phase * 7U + (token * 13U + token / 5U) % 24U) % expertCount);
        const auto found = positions.find(expert);
        if (found != positions.end()) {
            ++result.hits;
            resident.splice(resident.begin(), resident, found->second);
            found->second = resident.begin();
            continue;
        }
        result.transferredBytes += expertBytes;
        if (resident.size() == result.capacity && !resident.empty()) {
            positions.erase(resident.back());
            resident.pop_back();
        }
        if (result.capacity != 0) {
            resident.push_front(expert);
            positions[expert] = resident.begin();
        }
    }
    return result;
}

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double hitRate(const CacheResult& value) {
    return value.requests == 0
        ? 0.0
        : static_cast<double>(value.hits) /
              static_cast<double>(value.requests);
}

} // namespace

int main(int argc, char** argv) {
    try {
        using namespace hypermoe;
        using namespace hypermoe::tensor;
        constexpr std::size_t hidden = 128;
        constexpr std::size_t intermediate = 256;
        constexpr std::size_t iterations = 40;
        constexpr std::size_t comparisonBudget = 2U * 1024U * 1024U;
        const std::filesystem::path reportPath = argc > 1
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("phase22_quantized_expert_report.json");

        auto backend = std::make_shared<CpuTensorBackend>();
        auto input = backend->allocateTensor({1, hidden}, DType::FP32);
        auto gate = backend->allocateTensor({hidden, intermediate}, DType::FP32);
        auto up = backend->allocateTensor({hidden, intermediate}, DType::FP32);
        auto down = backend->allocateTensor({intermediate, hidden}, DType::FP32);
        auto fp32Output = backend->allocateTensor({1, hidden}, DType::FP32);
        auto int8Output = backend->allocateTensor({1, hidden}, DType::FP32);
        auto* inputValues = static_cast<float*>(input.data());
        auto fill = [](Tensor& tensor, float phase) {
            auto* values = static_cast<float*>(tensor.data());
            for (std::size_t index = 0; index < tensor.shape().elementCount(); ++index) {
                values[index] = 0.25F * std::sin(
                    static_cast<float>(index) * 0.013F + phase);
            }
        };
        for (std::size_t index = 0; index < hidden; ++index) {
            inputValues[index] = std::cos(static_cast<float>(index) * 0.031F);
        }
        fill(gate, 0.1F);
        fill(up, 0.7F);
        fill(down, 1.3F);

        const auto quantizeTensor = [&](const Tensor& source) {
            const auto values = std::span<const float>(
                static_cast<const float*>(source.data()),
                source.shape().elementCount());
            return quantization::quantizeInt8(values);
        };
        const auto quantizeStart = Clock::now();
        const auto gateInt8 = quantizeTensor(gate);
        const auto upInt8 = quantizeTensor(up);
        const auto downInt8 = quantizeTensor(down);
        const auto quantizeEnd = Clock::now();
        auto gateStorage = backend->allocateTensor(gate.shape(), DType::INT8);
        auto upStorage = backend->allocateTensor(up.shape(), DType::INT8);
        auto downStorage = backend->allocateTensor(down.shape(), DType::INT8);
        std::memcpy(gateStorage.data(), gateInt8.bytes.data(), gateInt8.bytes.size());
        std::memcpy(upStorage.data(), upInt8.bytes.data(), upInt8.bytes.size());
        std::memcpy(downStorage.data(), downInt8.bytes.data(), downInt8.bytes.size());

        ExpertMlpExecutor executor(backend);
        const ExpertMlpWeights fp32Weights{gate.view(), up.view(), down.view()};
        const ExpertMlpWeights int8Weights{
            gateStorage.view(), upStorage.view(), downStorage.view(),
            gateInt8.parameters, upInt8.parameters, downInt8.parameters};
        const auto fp32FirstStart = Clock::now();
        executor.execute(input.view(), fp32Weights, fp32Output.view());
        const auto fp32FirstEnd = Clock::now();
        const auto int8FirstStart = Clock::now();
        executor.execute(input.view(), int8Weights, int8Output.view());
        const auto int8FirstEnd = Clock::now();

        const auto fp32Start = Clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            executor.execute(input.view(), fp32Weights, fp32Output.view());
        }
        const auto fp32End = Clock::now();
        const auto int8Start = Clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            executor.execute(input.view(), int8Weights, int8Output.view());
        }
        const auto int8End = Clock::now();

        const auto* reference = static_cast<const float*>(fp32Output.data());
        const auto* actual = static_cast<const float*>(int8Output.data());
        double meanError{};
        double maximumError{};
        for (std::size_t index = 0; index < hidden; ++index) {
            const auto error = std::fabs(
                static_cast<double>(reference[index]) - actual[index]);
            meanError += error;
            maximumError = std::max(maximumError, error);
        }
        meanError /= static_cast<double>(hidden);

        const auto fp32ExpertBytes = gate.bytes() + up.bytes() + down.bytes();
        const auto int8ExpertBytes =
            gateStorage.bytes() + upStorage.bytes() + downStorage.bytes();
        const auto fp32Cache = simulateCache(fp32ExpertBytes, comparisonBudget);
        const auto int8Cache = simulateCache(int8ExpertBytes, comparisonBudget);
        const auto fp32Elapsed = milliseconds(fp32Start, fp32End);
        const auto int8Elapsed = milliseconds(int8Start, int8End);

        std::ofstream report(reportPath, std::ios::trunc);
        report << std::fixed << std::setprecision(6)
               << "{\n"
               << "  \"schema\": \"hypermoe.phase22a-benchmark.v1\",\n"
               << "  \"scope\": \"deterministic_cpu_expert_fixture\",\n"
               << "  \"hidden_dimension\": " << hidden << ",\n"
               << "  \"intermediate_dimension\": " << intermediate << ",\n"
               << "  \"comparison_budget_bytes\": " << comparisonBudget << ",\n"
               << "  \"quantization_time_ms\": "
               << milliseconds(quantizeStart, quantizeEnd) << ",\n"
               << "  \"before\": {\n"
               << "    \"expert_bytes\": " << fp32ExpertBytes << ",\n"
               << "    \"resident_capacity\": " << fp32Cache.capacity << ",\n"
               << "    \"transfer_bytes\": " << fp32Cache.transferredBytes << ",\n"
               << "    \"cache_hit_rate\": " << hitRate(fp32Cache) << ",\n"
               << "    \"first_token_latency_ms\": "
               << milliseconds(fp32FirstStart, fp32FirstEnd) << ",\n"
               << "    \"tokens_per_second\": "
               << static_cast<double>(iterations) * 1000.0 / fp32Elapsed << "\n"
               << "  },\n"
               << "  \"after\": {\n"
               << "    \"expert_bytes\": " << int8ExpertBytes << ",\n"
               << "    \"resident_capacity\": " << int8Cache.capacity << ",\n"
               << "    \"transfer_bytes\": " << int8Cache.transferredBytes << ",\n"
               << "    \"cache_hit_rate\": " << hitRate(int8Cache) << ",\n"
               << "    \"first_token_latency_ms\": "
               << milliseconds(int8FirstStart, int8FirstEnd) << ",\n"
               << "    \"tokens_per_second\": "
               << static_cast<double>(iterations) * 1000.0 / int8Elapsed << "\n"
               << "  },\n"
               << "  \"artifact_size_reduction_ratio\": "
               << static_cast<double>(fp32ExpertBytes) /
                      static_cast<double>(int8ExpertBytes) << ",\n"
               << "  \"mean_absolute_output_error\": " << meanError << ",\n"
               << "  \"maximum_absolute_output_error\": " << maximumError << "\n"
               << "}\n";
        if (!report) throw std::runtime_error("failed writing Phase 22 report");
        std::cout << reportPath << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase 22 benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
