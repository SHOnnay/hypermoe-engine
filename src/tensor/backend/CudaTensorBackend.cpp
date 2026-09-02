#include "tensor/backend/CudaTensorBackend.hpp"

#include "backend/CudaBackend.hpp"
#include "backend/cuda/CudaMemoryPool.hpp"
#include "backend/cuda/CudaRuntime.hpp"
#include "backend/cuda/CudaStreamManager.hpp"
#include "profiling/Profiler.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef HYPERMOE_HAS_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace hypermoe::tensor {

struct CudaTensorBackend::Impl {
    explicit Impl(int selectedDevice, std::shared_ptr<Profiler> selectedProfiler)
        : ordinal(selectedDevice), profiler(std::move(selectedProfiler)) {}

    ~Impl() {
#ifdef HYPERMOE_HAS_CUBLAS
        if (handle != nullptr) (void)cublasDestroy(handle);
#endif
    }

    int ordinal{};
    std::shared_ptr<Profiler> profiler;
    std::shared_ptr<backend::CudaRuntime> runtime;
    std::shared_ptr<backend::CudaBackend> backend;
    std::shared_ptr<backend::CudaMemoryPool> pool;
    std::unique_ptr<backend::CudaStreamManager> streams;
    std::string backendName{"CUDA tensor backend unavailable"};
    bool ready{};
#ifdef HYPERMOE_HAS_CUBLAS
    cublasHandle_t handle{};
#endif
};

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

void validateCopy(const Tensor& source, const Tensor& destination, int ordinal) {
    if (!source || !destination || !source.isContiguous() ||
        !destination.isContiguous()) {
        throw std::invalid_argument("CUDA tensor copy requires contiguous tensors");
    }
    if (source.dtype() != destination.dtype() || source.shape() != destination.shape()) {
        throw std::invalid_argument("CUDA tensor copy metadata mismatch");
    }
    if ((source.device().type == DeviceType::CUDA &&
         source.device().ordinal != ordinal) ||
        (destination.device().type == DeviceType::CUDA &&
         destination.device().ordinal != ordinal)) {
        throw std::invalid_argument("CUDA tensor belongs to another device");
    }
}

void validateCudaElementwise(const Tensor& left,
                             const Tensor& right,
                             const Tensor& output,
                             int ordinal,
                             const char* operation) {
    if (!left || !right || !output || left.device() != Device::cuda(ordinal) ||
        right.device() != Device::cuda(ordinal) ||
        output.device() != Device::cuda(ordinal) || !left.isContiguous() ||
        !right.isContiguous() || !output.isContiguous() ||
        left.dtype() != DType::FP32 || right.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || left.shape() != right.shape() ||
        left.shape() != output.shape()) {
        throw std::invalid_argument(std::string(operation) +
                                    " requires equal contiguous CUDA FP32 tensors");
    }
}

#ifdef HYPERMOE_HAS_CUBLAS
void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
}
#endif

} // namespace

CudaTensorBackend::CudaTensorBackend(int device,
                                     std::shared_ptr<Profiler> profiler)
    : impl_(std::make_unique<Impl>(device, std::move(profiler))) {
    impl_->runtime = std::make_shared<backend::CudaRuntime>(device);
    if (!impl_->runtime->available()) return;
#ifdef HYPERMOE_HAS_CUBLAS
    impl_->backend = std::make_shared<backend::CudaBackend>(device);
    impl_->pool = std::make_shared<backend::CudaMemoryPool>(impl_->backend);
    impl_->streams =
        std::make_unique<backend::CudaStreamManager>(impl_->runtime);
    checkCublas(cublasCreate(&impl_->handle), "cublasCreate");
    checkCublas(cublasSetStream(
                    impl_->handle,
                    reinterpret_cast<cudaStream_t>(impl_->streams->stream(
                        backend::CudaStreamRole::Compute))),
                "cublasSetStream");
    impl_->backendName = "CUDA cuBLAS tensor backend";
    impl_->ready = true;
#endif
}

CudaTensorBackend::~CudaTensorBackend() {
    if (impl_ && impl_->ready) {
        try {
            synchronize();
        } catch (...) {
            // Destructors cannot propagate device shutdown failures.
        }
    }
}

std::string_view CudaTensorBackend::name() const noexcept {
    return impl_->backendName;
}

Device CudaTensorBackend::device() const noexcept {
    return Device::cuda(impl_->ordinal);
}

bool CudaTensorBackend::available() const noexcept { return impl_->ready; }

Tensor CudaTensorBackend::allocateTensor(const Shape& shape, DType dtype) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    auto buffer = impl_->pool->allocateDeviceBuffer(tensorStorageBytes(shape, dtype));
    auto tensor = Tensor::fromDeviceBuffer(shape, dtype, device(), std::move(buffer));
    if (impl_->profiler) impl_->profiler->recordTensorAllocation();
    return tensor;
}

void CudaTensorBackend::copyTensor(const Tensor& source, Tensor& destination) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    validateCopy(source, destination, impl_->ordinal);
    if (source.device().type == DeviceType::CPU &&
        destination.device().type == DeviceType::CPU) {
        std::memmove(destination.data(), source.data(), source.bytes());
        return;
    }

    const auto stream =
        impl_->streams->stream(backend::CudaStreamRole::Transfer);
    auto completion = impl_->backend->createEvent();
    try {
        if (source.device().type == DeviceType::CPU) {
            impl_->backend->copyToDevice(destination.data(), source.data(),
                                         source.bytes(), stream);
        } else if (destination.device().type == DeviceType::CPU) {
            impl_->backend->copyFromDevice(destination.data(), source.data(),
                                           source.bytes(), stream);
        } else {
#ifdef HYPERMOE_HAS_CUBLAS
            checkCuda(cudaSetDevice(impl_->ordinal), "cudaSetDevice");
            checkCuda(cudaMemcpyAsync(destination.data(), source.data(), source.bytes(),
                                      cudaMemcpyDeviceToDevice, stream),
                      "cudaMemcpyAsync(device-to-device)");
#else
            throw std::runtime_error("CUDA device copy support is unavailable");
#endif
        }
        impl_->backend->recordEvent(completion, stream);
        impl_->backend->waitEvent(completion);
        impl_->backend->synchronize(stream);
        impl_->backend->destroyEvent(completion);
    } catch (...) {
        impl_->backend->destroyEvent(completion);
        throw;
    }
}

void CudaTensorBackend::matmul(const Tensor& left,
                               const Tensor& right,
                               Tensor& output) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    if (!left || !right || !output || left.device() != device() ||
        right.device() != device() || output.device() != device() ||
        !left.isContiguous() || !right.isContiguous() || !output.isContiguous() ||
        left.dtype() != DType::FP32 || right.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || left.shape().rank() != 2 ||
        right.shape().rank() != 2 || output.shape().rank() != 2) {
        throw std::invalid_argument(
            "CUDA matmul requires contiguous rank-2 CUDA FP32 tensors");
    }
    const auto& leftDims = left.shape().dimensions();
    const auto& rightDims = right.shape().dimensions();
    const auto& outputDims = output.shape().dimensions();
    if (rightDims[0] != leftDims[1] || outputDims[0] != leftDims[0] ||
        outputDims[1] != rightDims[1]) {
        throw std::invalid_argument("CUDA matmul dimensions are incompatible");
    }
    if (left.data() == output.data() || right.data() == output.data()) {
        throw std::invalid_argument("CUDA matmul output cannot alias an input");
    }
    if (leftDims[0] > static_cast<std::size_t>(INT_MAX) ||
        leftDims[1] > static_cast<std::size_t>(INT_MAX) ||
        rightDims[1] > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error("CUDA matmul dimensions exceed cuBLAS integer limits");
    }

#ifdef HYPERMOE_HAS_CUBLAS
    const auto rows = static_cast<int>(leftDims[0]);
    const auto inner = static_cast<int>(leftDims[1]);
    const auto columns = static_cast<int>(rightDims[1]);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto stream =
        impl_->streams->stream(backend::CudaStreamRole::Compute);
    auto eventStart = impl_->runtime->createEvent(true);
    backend::EventHandle eventEnd = nullptr;
    try {
        eventEnd = impl_->runtime->createEvent(true);
        impl_->runtime->recordEvent(eventStart, stream);
        checkCublas(cublasSgemm(
                        impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N, columns, rows,
                        inner, &alpha, static_cast<const float*>(right.data()),
                        columns, static_cast<const float*>(left.data()), inner,
                        &beta, static_cast<float*>(output.data()), columns),
                    "cublasSgemm");
        impl_->runtime->recordEvent(eventEnd, stream);
        impl_->runtime->synchronizeEvent(eventEnd);
        const auto milliseconds =
            static_cast<double>(impl_->runtime->elapsedMilliseconds(eventStart, eventEnd));
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double, std::milli>(milliseconds));
        if (impl_->profiler) {
            impl_->profiler->recordKernelTime(elapsed);
            impl_->profiler->recordMatmulTime(elapsed);
        }
        impl_->runtime->destroyEvent(eventStart);
        impl_->runtime->destroyEvent(eventEnd);
    } catch (...) {
        impl_->runtime->destroyEvent(eventStart);
        impl_->runtime->destroyEvent(eventEnd);
        throw;
    }
#else
    (void)left;
    (void)right;
    (void)output;
    throw std::runtime_error("cuBLAS support is unavailable");
#endif
}

void CudaTensorBackend::add(const Tensor& left,
                            const Tensor& right,
                            Tensor& output) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    validateCudaElementwise(left, right, output, impl_->ordinal, "CUDA add");
    CpuTensorBackend cpu(impl_->profiler);
    auto hostLeft = cpu.allocateTensor(left.shape(), left.dtype());
    auto hostRight = cpu.allocateTensor(right.shape(), right.dtype());
    auto hostOutput = cpu.allocateTensor(output.shape(), output.dtype());
    copyTensor(left, hostLeft);
    copyTensor(right, hostRight);
    cpu.add(hostLeft, hostRight, hostOutput);
    copyTensor(hostOutput, output);
}

void CudaTensorBackend::mul(const Tensor& left,
                            const Tensor& right,
                            Tensor& output) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    validateCudaElementwise(left, right, output, impl_->ordinal, "CUDA multiply");
    CpuTensorBackend cpu(impl_->profiler);
    auto hostLeft = cpu.allocateTensor(left.shape(), left.dtype());
    auto hostRight = cpu.allocateTensor(right.shape(), right.dtype());
    auto hostOutput = cpu.allocateTensor(output.shape(), output.dtype());
    copyTensor(left, hostLeft);
    copyTensor(right, hostRight);
    cpu.mul(hostLeft, hostRight, hostOutput);
    copyTensor(hostOutput, output);
}

Tensor CudaTensorBackend::reshape(const Tensor& tensor, Shape shape) {
    if (tensor.device() != device()) {
        throw std::invalid_argument("CUDA reshape received a tensor from another device");
    }
    return tensor.reshape(std::move(shape));
}

void CudaTensorBackend::synchronize() {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    impl_->backend->synchronize(
        impl_->streams->stream(backend::CudaStreamRole::Compute));
    impl_->backend->synchronize(
        impl_->streams->stream(backend::CudaStreamRole::Transfer));
    impl_->backend->synchronize(
        impl_->streams->stream(backend::CudaStreamRole::Prefetch));
}

} // namespace hypermoe::tensor
