#include "tensor/activation/Activation.hpp"

#include "profiling/Profiler.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace hypermoe::tensor::activation {
namespace {

void validate(TensorView input, TensorView output, Device expectedDevice) {
    if (!input || !output || input.device() != expectedDevice ||
        output.device() != expectedDevice || input.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || input.shape() != output.shape() ||
        !input.isContiguous() || !output.isContiguous() || !output.writable()) {
        throw std::invalid_argument(
            "activation requires equal contiguous writable-output FP32 tensors");
    }
}

void validateType(ActivationType type) {
    switch (type) {
    case ActivationType::SiLU:
    case ActivationType::GELU: return;
    }
    throw std::invalid_argument("unsupported activation type");
}

void applyCpu(ActivationType type, TensorView input, TensorView output) {
    const auto* source = static_cast<const float*>(input.data());
    auto* destination = static_cast<float*>(output.mutableData());
    for (std::size_t index = 0; index < input.shape().elementCount(); ++index) {
        destination[index] = type == ActivationType::SiLU ? silu(source[index])
                                                          : gelu(source[index]);
    }
}

} // namespace

float silu(float value) noexcept { return value / (1.0F + std::exp(-value)); }

float gelu(float value) noexcept {
    constexpr float inverseSqrtTwo = 0.70710678118654752440F;
    return 0.5F * value * (1.0F + std::erf(value * inverseSqrtTwo));
}

void apply(ActivationType type,
           TensorBackend& backend,
           TensorView input,
           TensorView output,
           const std::shared_ptr<Profiler>& profiler) {
    validateType(type);
    [[maybe_unused]] const auto inputOwner = input.lockOwner();
    [[maybe_unused]] const auto outputOwner = output.lockOwner();
    if (!inputOwner || !outputOwner) {
        throw std::invalid_argument("activation received expired tensor storage");
    }
    validate(input, output, backend.device());
    const auto start = std::chrono::steady_clock::now();
    if (backend.device().type == DeviceType::CPU) {
        applyCpu(type, input, output);
    } else {
        CpuTensorBackend cpu;
        auto hostInput = cpu.allocateTensor(input.shape(), input.dtype());
        auto hostOutput = cpu.allocateTensor(output.shape(), output.dtype());
        backend.copyTensor(input, hostInput);
        applyCpu(type, hostInput, hostOutput);
        backend.copyTensor(hostOutput, output);
    }
    if (profiler) {
        profiler->recordActivationTime(std::chrono::steady_clock::now() - start);
    }
}

} // namespace hypermoe::tensor::activation
