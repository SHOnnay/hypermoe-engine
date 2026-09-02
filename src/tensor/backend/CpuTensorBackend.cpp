#include "tensor/backend/CpuTensorBackend.hpp"

#include "backend/CpuBackend.hpp"
#include "profiling/Profiler.hpp"

#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypermoe::tensor {
namespace {

std::size_t tensorStorageBytes(const Shape& shape, DType dtype) {
    const auto elementBytes = sizeOf(dtype);
    if (elementBytes == 0 ||
        shape.storageElementCount() >
            std::numeric_limits<std::size_t>::max() / elementBytes) {
        throw std::overflow_error("tensor allocation byte size overflow");
    }
    return shape.storageElementCount() * elementBytes;
}

void validateCpuContiguous(TensorView tensor, const char* operation) {
    if (!tensor || tensor.device().type != DeviceType::CPU || !tensor.isContiguous()) {
        throw std::invalid_argument(std::string(operation) +
                                    " requires a contiguous CPU tensor");
    }
}

std::shared_ptr<void> pin(TensorView tensor, const char* operation) {
    auto owner = tensor.lockOwner();
    if (!owner) {
        throw std::invalid_argument(std::string(operation) +
                                    " received expired tensor storage");
    }
    return owner;
}

void validateElementwise(TensorView left,
                         TensorView right,
                         TensorView output,
                         const char* operation) {
    validateCpuContiguous(left, operation);
    validateCpuContiguous(right, operation);
    validateCpuContiguous(output, operation);
    if (left.dtype() != DType::FP32 || right.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || left.shape() != right.shape() ||
        left.shape() != output.shape() || !output.writable()) {
        throw std::invalid_argument(std::string(operation) +
                                    " requires equal-shape FP32 tensors");
    }
}

} // namespace

CpuTensorBackend::CpuTensorBackend(std::shared_ptr<Profiler> profiler)
    : backend_(std::make_shared<backend::CpuBackend>()),
      profiler_(std::move(profiler)) {}

std::string_view CpuTensorBackend::name() const noexcept { return "CPU tensor backend"; }
Device CpuTensorBackend::device() const noexcept { return Device::cpu(); }
bool CpuTensorBackend::available() const noexcept { return true; }

Tensor CpuTensorBackend::allocateTensor(const Shape& shape, DType dtype) {
    auto buffer = std::make_shared<backend::DeviceBuffer>(
        backend_, tensorStorageBytes(shape, dtype));
    auto tensor = Tensor::fromDeviceBuffer(shape, dtype, device(), std::move(buffer));
    if (profiler_) profiler_->recordTensorAllocation();
    return tensor;
}

void CpuTensorBackend::copyTensor(TensorView source, TensorView destination) {
    [[maybe_unused]] const auto sourceOwner = pin(source, "CPU tensor copy");
    [[maybe_unused]] const auto destinationOwner = pin(destination, "CPU tensor copy");
    validateCpuContiguous(source, "CPU tensor copy");
    validateCpuContiguous(destination, "CPU tensor copy");
    if (source.dtype() != destination.dtype() ||
        source.shape() != destination.shape() || !destination.writable()) {
        throw std::invalid_argument("CPU tensor copy metadata mismatch");
    }
    std::memmove(destination.mutableData(), source.data(), source.bytes());
}

void CpuTensorBackend::matmul(TensorView left,
                              TensorView right,
                              TensorView output) {
    [[maybe_unused]] const auto leftOwner = pin(left, "CPU matmul");
    [[maybe_unused]] const auto rightOwner = pin(right, "CPU matmul");
    [[maybe_unused]] const auto outputOwner = pin(output, "CPU matmul");
    validateCpuContiguous(left, "CPU matmul");
    validateCpuContiguous(right, "CPU matmul");
    validateCpuContiguous(output, "CPU matmul");
    if (left.dtype() != DType::FP32 || right.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || left.shape().rank() != 2 ||
        right.shape().rank() != 2 || output.shape().rank() != 2 ||
        !output.writable()) {
        throw std::invalid_argument("CPU matmul requires contiguous rank-2 FP32 tensors");
    }
    const auto& leftDims = left.shape().dimensions();
    const auto& rightDims = right.shape().dimensions();
    const auto& outputDims = output.shape().dimensions();
    const auto rows = leftDims[0];
    const auto inner = leftDims[1];
    const auto columns = rightDims[1];
    if (rightDims[0] != inner || outputDims[0] != rows ||
        outputDims[1] != columns) {
        throw std::invalid_argument("CPU matmul dimensions are incompatible");
    }

    const auto start = std::chrono::steady_clock::now();
    const auto* leftData = static_cast<const float*>(left.data());
    const auto* rightData = static_cast<const float*>(right.data());
    auto* outputData = static_cast<float*>(output.mutableData());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::size_t index = 0; index < inner; ++index) {
                sum += leftData[row * inner + index] *
                       rightData[index * columns + column];
            }
            outputData[row * columns + column] = sum;
        }
    }
    if (profiler_) profiler_->recordMatmulTime(std::chrono::steady_clock::now() - start);
}

void CpuTensorBackend::add(TensorView left,
                           TensorView right,
                           TensorView output) {
    [[maybe_unused]] const auto leftOwner = pin(left, "CPU add");
    [[maybe_unused]] const auto rightOwner = pin(right, "CPU add");
    [[maybe_unused]] const auto outputOwner = pin(output, "CPU add");
    validateElementwise(left, right, output, "CPU add");
    const auto* leftData = static_cast<const float*>(left.data());
    const auto* rightData = static_cast<const float*>(right.data());
    auto* outputData = static_cast<float*>(output.mutableData());
    for (std::size_t index = 0; index < left.shape().elementCount(); ++index) {
        outputData[index] = leftData[index] + rightData[index];
    }
}

void CpuTensorBackend::mul(TensorView left,
                           TensorView right,
                           TensorView output) {
    [[maybe_unused]] const auto leftOwner = pin(left, "CPU multiply");
    [[maybe_unused]] const auto rightOwner = pin(right, "CPU multiply");
    [[maybe_unused]] const auto outputOwner = pin(output, "CPU multiply");
    validateElementwise(left, right, output, "CPU multiply");
    const auto* leftData = static_cast<const float*>(left.data());
    const auto* rightData = static_cast<const float*>(right.data());
    auto* outputData = static_cast<float*>(output.mutableData());
    for (std::size_t index = 0; index < left.shape().elementCount(); ++index) {
        outputData[index] = leftData[index] * rightData[index];
    }
}

Tensor CpuTensorBackend::reshape(const Tensor& tensor, Shape shape) {
    return tensor.reshape(std::move(shape));
}

void CpuTensorBackend::synchronize() { backend_->synchronize(); }

} // namespace hypermoe::tensor
