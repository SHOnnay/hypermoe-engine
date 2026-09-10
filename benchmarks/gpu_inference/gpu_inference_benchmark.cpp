#include "backend/CudaBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "transformer/attention/CudaAttention.hpp"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration value) {
    return std::chrono::duration<double, std::milli>(value).count();
}

void fill(hypermoe::tensor::Tensor& tensor, float seed) {
    auto* values = static_cast<float*>(tensor.data());
    for (std::size_t index = 0; index < tensor.shape().elementCount(); ++index) {
        values[index] = seed + static_cast<float>(index % 23U) * 0.001F;
    }
}

std::string number(const std::optional<double>& value) {
    if (!value) return "null";
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << *value;
    return output.str();
}

struct Report {
    double cpuExpertMs{};
    bool cudaAvailable{};
    std::string device;
    std::optional<double> initializationMs;
    std::optional<double> hostToDeviceMs;
    std::optional<double> cudaExpertMs;
    std::optional<double> cudaAttentionMs;
    std::optional<double> transformerLayerMs;
    std::optional<double> tokensPerSecond;
    std::optional<double> allocatedMiB;

    [[nodiscard]] std::string json() const {
        std::ostringstream output;
        output << "{\n"
               << "  \"schema\": \"hypermoe.gpu-inference.v1\",\n"
               << "  \"cuda_available\": " << (cudaAvailable ? "true" : "false") << ",\n"
               << "  \"device\": \"" << device << "\",\n"
               << "  \"gpu_initialization_ms\": " << number(initializationMs) << ",\n"
               << "  \"h2d_transfer_ms\": " << number(hostToDeviceMs) << ",\n"
               << "  \"cpu_expert_ms\": " << number(cpuExpertMs) << ",\n"
               << "  \"cuda_expert_ms\": " << number(cudaExpertMs) << ",\n"
               << "  \"cuda_attention_ms\": " << number(cudaAttentionMs) << ",\n"
               << "  \"transformer_layer_ms\": " << number(transformerLayerMs) << ",\n"
               << "  \"tokens_per_second\": " << number(tokensPerSecond) << ",\n"
               << "  \"vram_allocated_mib\": " << number(allocatedMiB) << "\n"
               << "}\n";
        return output.str();
    }
};

Report run() {
    using namespace hypermoe;
    constexpr std::size_t tokens = 4;
    constexpr std::size_t hidden = 128;
    constexpr std::size_t intermediate = 256;
    constexpr std::size_t iterations = 5;
    tensor::CpuTensorBackend cpu;
    auto input = cpu.allocateTensor({tokens, hidden}, tensor::DType::FP32);
    auto gate = cpu.allocateTensor({hidden, intermediate}, tensor::DType::FP32);
    auto up = cpu.allocateTensor({hidden, intermediate}, tensor::DType::FP32);
    auto down = cpu.allocateTensor({intermediate, hidden}, tensor::DType::FP32);
    auto output = cpu.allocateTensor({tokens, hidden}, tensor::DType::FP32);
    fill(input, 0.01F);
    fill(gate, 0.02F);
    fill(up, 0.03F);
    fill(down, 0.04F);
    ExpertMlpExecutor cpuExpert(std::make_shared<tensor::CpuTensorBackend>());
    auto start = Clock::now();
    for (std::size_t index = 0; index < iterations; ++index) {
        cpuExpert.execute(input.view(), {gate.view(), up.view(), down.view()},
                          output.view());
    }
    Report report;
    report.cpuExpertMs = milliseconds(Clock::now() - start) /
                         static_cast<double>(iterations);

    start = Clock::now();
    auto cuda = std::make_shared<tensor::CudaTensorBackend>();
    const auto initialization = Clock::now() - start;
    if (!cuda->available()) {
        report.device = "CPU fallback (CUDA unavailable)";
        return report;
    }
    report.cudaAvailable = true;
    report.initializationMs = milliseconds(initialization);
    auto raw = std::make_shared<backend::CudaBackend>();
    report.device = std::string(raw->name());

    auto deviceInput = cuda->allocateTensor(input.shape(), input.dtype());
    auto deviceGate = cuda->allocateTensor(gate.shape(), gate.dtype());
    auto deviceUp = cuda->allocateTensor(up.shape(), up.dtype());
    auto deviceDown = cuda->allocateTensor(down.shape(), down.dtype());
    auto deviceOutput = cuda->allocateTensor(output.shape(), output.dtype());
    start = Clock::now();
    cuda->copyTensor(input.view(), deviceInput.view());
    cuda->copyTensor(gate.view(), deviceGate.view());
    cuda->copyTensor(up.view(), deviceUp.view());
    cuda->copyTensor(down.view(), deviceDown.view());
    report.hostToDeviceMs = milliseconds(Clock::now() - start);
    ExpertMlpExecutor cudaExpert(cuda);
    start = Clock::now();
    for (std::size_t index = 0; index < iterations; ++index) {
        cudaExpert.execute(deviceInput.view(),
                           {deviceGate.view(), deviceUp.view(), deviceDown.view()},
                           deviceOutput.view());
    }
    report.cudaExpertMs = milliseconds(Clock::now() - start) /
                          static_cast<double>(iterations);

    auto identityHost = cpu.allocateTensor({hidden, hidden}, tensor::DType::FP32);
    std::memset(identityHost.data(), 0, identityHost.bytes());
    auto* identity = static_cast<float*>(identityHost.data());
    for (std::size_t index = 0; index < hidden; ++index) {
        identity[index * hidden + index] = 1.0F;
    }
    auto identityDevice = cuda->allocateTensor(identityHost.shape(), identityHost.dtype());
    cuda->copyTensor(identityHost.view(), identityDevice.view());
    transformer::attention::CudaAttention attention(cuda);
    transformer::attention::AttentionConfiguration configuration;
    configuration.headCount = 4;
    configuration.keyValueHeadCount = 4;
    configuration.headDimension = hidden / 4;
    configuration.causal = true;
    start = Clock::now();
    auto attentionResult = attention.execute(
        deviceInput.view(),
        {identityDevice.view(), identityDevice.view(), identityDevice.view(),
         identityDevice.view()}, configuration);
    report.cudaAttentionMs = milliseconds(Clock::now() - start);
    report.transformerLayerMs = *report.cudaAttentionMs + *report.cudaExpertMs;
    report.tokensPerSecond = static_cast<double>(tokens) /
        (*report.transformerLayerMs / 1000.0);
    report.allocatedMiB = static_cast<double>(raw->stats().allocatedBytes) /
                          (1024.0 * 1024.0);
    return report;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto report = run();
        const auto json = report.json();
        const std::string path = argc > 1 ? argv[1] : "gpu_inference_report.json";
        std::ofstream output(path, std::ios::trunc);
        output << json;
        if (!output) throw std::runtime_error("failed writing GPU inference report");
        std::cout << json;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU inference benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
