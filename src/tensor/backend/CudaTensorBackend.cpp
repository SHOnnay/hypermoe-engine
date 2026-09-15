#include "tensor/backend/CudaTensorBackend.hpp"

#include "backend/CudaBackend.hpp"
#include "backend/cuda/CudaMemoryPool.hpp"
#include "backend/cuda/CudaRuntime.hpp"
#include "backend/cuda/CudaStreamManager.hpp"
#ifdef HYPERMOE_HAS_CUDA_KERNELS
#include "backend/cuda/CudaKernels.hpp"
#endif
#include "profiling/Profiler.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
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
        events.reset(); // Complete retained storage before streams/pool/cuBLAS teardown.
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
    std::unique_ptr<profiling::GpuEventQueue> events;
    backend::cuda::Int8GemmMode int8Mode{backend::cuda::Int8GemmMode::Auto};
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

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const char* operation) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(operation) + " size overflow");
    }
    return left * right;
}

void validateCopy(TensorView source, TensorView destination, int ordinal) {
    if (!source || !destination || !source.isContiguous() ||
        !destination.isContiguous()) {
        throw std::invalid_argument("CUDA tensor copy requires contiguous tensors");
    }
    if (source.dtype() != destination.dtype() || source.shape() != destination.shape() ||
        !destination.writable()) {
        throw std::invalid_argument("CUDA tensor copy metadata mismatch");
    }
    if ((source.device().type == DeviceType::CUDA &&
         source.device().ordinal != ordinal) ||
        (destination.device().type == DeviceType::CUDA &&
         destination.device().ordinal != ordinal)) {
        throw std::invalid_argument("CUDA tensor belongs to another device");
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

void validateCudaElementwise(TensorView left,
                             TensorView right,
                             TensorView output,
                             int ordinal,
                             const char* operation) {
    if (!left || !right || !output || left.device() != Device::cuda(ordinal) ||
        right.device() != Device::cuda(ordinal) ||
        output.device() != Device::cuda(ordinal) || !left.isContiguous() ||
        !right.isContiguous() || !output.isContiguous() ||
        left.dtype() != DType::FP32 || right.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || left.shape() != right.shape() ||
        left.shape() != output.shape() || !output.writable()) {
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
                                     std::shared_ptr<Profiler> profiler,
                                     backend::cuda::Int8GemmMode int8Mode)
    : impl_(std::make_unique<Impl>(device, std::move(profiler))) {
    impl_->runtime = std::make_shared<backend::CudaRuntime>(device);
    // Validate selection even in CPU-only builds.
    (void)backend::cuda::planInt8Gemm(1, 1, 1, int8Mode);
    impl_->int8Mode = int8Mode;
    if (!impl_->runtime->available()) return;
#ifdef HYPERMOE_HAS_CUBLAS
    impl_->backend = std::make_shared<backend::CudaBackend>(device);
    impl_->pool = std::make_shared<backend::CudaMemoryPool>(impl_->backend);
    impl_->streams =
        std::make_unique<backend::CudaStreamManager>(impl_->runtime);
    const auto runtime = impl_->runtime;
    const auto memoryBackend = impl_->backend;
    impl_->events = std::make_unique<profiling::GpuEventQueue>(profiling::GpuEventApi{
        [runtime](bool timing) { return runtime->createEvent(timing); },
        [runtime](backend::EventHandle event, backend::StreamHandle stream) { runtime->recordEvent(event, stream); },
        [runtime](backend::EventHandle event) { return runtime->eventComplete(event); },
        [runtime](backend::EventHandle start, backend::EventHandle end) { return runtime->elapsedMilliseconds(start, end); },
        [runtime](backend::EventHandle event) { runtime->destroyEvent(event); },
        [memoryBackend](backend::StreamHandle stream) { memoryBackend->synchronize(stream); }
    }, impl_->profiler);
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

bool CudaTensorBackend::nativeKernelsAvailable() const noexcept {
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    return available();
#else
    return false;
#endif
}

backend::BackendStats CudaTensorBackend::backendStats() const {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    impl_->events->collect();
    return impl_->backend->stats();
}

profiling::GpuEventQueue::Scope CudaTensorBackend::timeRegion(profiling::GpuOperation operation) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    impl_->events->collect();
    return impl_->events->begin(operation, impl_->streams->stream(backend::CudaStreamRole::Compute));
}

Tensor CudaTensorBackend::allocateTensor(const Shape& shape, DType dtype) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    auto buffer = impl_->pool->allocateDeviceBuffer(tensorStorageBytes(shape, dtype));
    auto tensor = Tensor::fromDeviceBuffer(shape, dtype, device(), std::move(buffer));
    if (impl_->profiler) impl_->profiler->recordTensorAllocation();
    return tensor;
}

void CudaTensorBackend::copyTensor(TensorView source, TensorView destination) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    [[maybe_unused]] const auto sourceOwner = pin(source, "CUDA tensor copy");
    [[maybe_unused]] const auto destinationOwner = pin(destination, "CUDA tensor copy");
    validateCopy(source, destination, impl_->ordinal);
    if (source.device().type == DeviceType::CPU &&
        destination.device().type == DeviceType::CPU) {
        std::memmove(destination.mutableData(), source.data(), source.bytes());
        return;
    }

    const auto stream =
        impl_->streams->stream(backend::CudaStreamRole::Transfer);
    backend::EventHandle sourceReady = nullptr;
    auto completion = impl_->backend->createEvent();
    try {
        // Both source reads and destination writes depend on prior compute.
        // Queue the dependency rather than blocking the host on that stream.
        sourceReady = impl_->runtime->createEvent(false);
        impl_->runtime->recordEvent(sourceReady,
            impl_->streams->stream(backend::CudaStreamRole::Compute));
        impl_->runtime->waitStreamEvent(stream, sourceReady);
        if (source.device().type == DeviceType::CPU) {
            impl_->backend->copyToDevice(destination.mutableData(), source.data(),
                                         source.bytes(), stream);
        } else if (destination.device().type == DeviceType::CPU) {
            impl_->backend->copyFromDevice(destination.mutableData(), source.data(),
                                           source.bytes(), stream);
        } else {
#ifdef HYPERMOE_HAS_CUBLAS
            checkCuda(cudaSetDevice(impl_->ordinal), "cudaSetDevice");
            checkCuda(cudaMemcpyAsync(destination.mutableData(), source.data(), source.bytes(),
                                      cudaMemcpyDeviceToDevice,
                                      static_cast<cudaStream_t>(stream)),
                      "cudaMemcpyAsync(device-to-device)");
#else
            throw std::runtime_error("CUDA device copy support is unavailable");
#endif
        }
        impl_->backend->recordEvent(completion, stream);
        impl_->backend->waitEvent(completion);
        impl_->backend->destroyEvent(completion);
        impl_->runtime->destroyEvent(sourceReady);
        impl_->events->collect();
    } catch (...) {
        // Exception-only lifetime barrier: raw host buffers must not escape in flight.
        try { impl_->backend->synchronize(stream); } catch (...) {}
        impl_->backend->destroyEvent(completion);
        impl_->runtime->destroyEvent(sourceReady);
        throw;
    }
}

void CudaTensorBackend::matmul(TensorView left,
                               TensorView right,
                               TensorView output) {
    matmulImpl(left, right, output, false);
}

void CudaTensorBackend::matmulExpert(TensorView left, TensorView right, TensorView output) {
    matmulImpl(left, right, output, true);
}

void CudaTensorBackend::matmulImpl(TensorView left, TensorView right, TensorView output,
                                  [[maybe_unused]] bool expert) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    [[maybe_unused]] const auto leftOwner = pin(left, "CUDA matmul");
    [[maybe_unused]] const auto rightOwner = pin(right, "CUDA matmul");
    [[maybe_unused]] const auto outputOwner = pin(output, "CUDA matmul");
    if (!left || !right || !output || left.device() != device() ||
        right.device() != device() || output.device() != device() ||
        !left.isContiguous() || !right.isContiguous() || !output.isContiguous() ||
        left.dtype() != DType::FP32 || right.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || left.shape().rank() != 2 ||
        right.shape().rank() != 2 || output.shape().rank() != 2 ||
        !output.writable()) {
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
    impl_->events->collect();
    auto operation = impl_->events->begin(expert ? profiling::GpuOperation::ExpertFp32Gemm :
        profiling::GpuOperation::Fp32Gemm, stream, {leftOwner, rightOwner, outputOwner});
    checkCublas(cublasSgemm(
                        impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N, columns, rows,
                        inner, &alpha, static_cast<const float*>(right.data()),
                        columns, static_cast<const float*>(left.data()), inner,
                        &beta, static_cast<float*>(output.mutableData()), columns),
                    "cublasSgemm");
    operation.finish();
#else
    (void)left;
    (void)right;
    (void)output;
    throw std::runtime_error("cuBLAS support is unavailable");
#endif
}

void CudaTensorBackend::matmulInt8Weights(
    TensorView left,
    TensorView right,
    const quantization::QuantizationParameters& parameters,
    TensorView output) {
    matmulInt8Impl(left, right, parameters, output, false);
}

void CudaTensorBackend::matmulInt8Expert(TensorView left, TensorView right,
    const quantization::QuantizationParameters& parameters, TensorView output) {
    matmulInt8Impl(left, right, parameters, output, true);
}

void CudaTensorBackend::matmulInt8Impl(TensorView left, TensorView right,
    const quantization::QuantizationParameters& parameters, TensorView output,
    [[maybe_unused]] bool expert) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    [[maybe_unused]] const auto leftOwner = pin(left, "CUDA INT8 weight matmul");
    [[maybe_unused]] const auto rightOwner = pin(right, "CUDA INT8 weight matmul");
    [[maybe_unused]] const auto outputOwner = pin(output, "CUDA INT8 weight matmul");
    quantization::validateParameters(quantization::QuantizedDType::INT8,
                                     parameters);
    if (!left || !right || !output || left.device() != device() ||
        right.device() != device() || output.device() != device() ||
        !left.isContiguous() || !right.isContiguous() ||
        !output.isContiguous() || left.dtype() != DType::FP32 ||
        right.dtype() != DType::INT8 || output.dtype() != DType::FP32 ||
        left.shape().rank() != 2 || right.shape().rank() != 2 ||
        output.shape().rank() != 2 || !output.writable()) {
        throw std::invalid_argument(
            "CUDA INT8 weight matmul requires FP32 input/output and rank-2 INT8 weights");
    }
    const auto& leftDims = left.shape().dimensions();
    const auto& rightDims = right.shape().dimensions();
    const auto& outputDims = output.shape().dimensions();
    if (rightDims[0] != leftDims[1] || outputDims[0] != leftDims[0] ||
        outputDims[1] != rightDims[1]) {
        throw std::invalid_argument(
            "CUDA INT8 weight matmul dimensions are incompatible");
    }
    if (left.data() == output.data() || right.data() == output.data()) {
        throw std::invalid_argument(
            "CUDA INT8 weight matmul output cannot alias an input");
    }
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    impl_->events->collect();
    const auto plan = backend::cuda::planInt8Gemm(leftDims[0], leftDims[1], rightDims[1], impl_->int8Mode);
    std::shared_ptr<backend::DeviceBuffer> scratch;
    if (plan.scratchElements != 0) scratch = impl_->pool->allocateDeviceBuffer(plan.scratchElements * sizeof(float));
    const auto stream = impl_->streams->stream(backend::CudaStreamRole::Compute);
    auto operation = impl_->events->begin(expert ? profiling::GpuOperation::ExpertInt8Gemm :
        profiling::GpuOperation::Int8Gemm, stream, {leftOwner, rightOwner, outputOwner, scratch});
    if (plan.implementation == backend::cuda::Int8GemmMode::Cooperative) {
        backend::cuda::kernels::int8WeightMatmulCooperative(
            static_cast<const float*>(left.data()), static_cast<const std::int8_t*>(right.data()),
            static_cast<float*>(output.mutableData()), scratch ? static_cast<float*>(scratch->data()) : nullptr,
            leftDims[0], leftDims[1], rightDims[1], plan.partitions, parameters.scale, parameters.zeroPoint, stream);
    } else {
        backend::cuda::kernels::int8WeightMatmul(
        static_cast<const float*>(left.data()),
        static_cast<const std::int8_t*>(right.data()),
        static_cast<float*>(output.mutableData()), leftDims[0], leftDims[1],
        rightDims[1], parameters.scale, parameters.zeroPoint,
        stream);
    }
    operation.finish();
#else
    (void)parameters;
    throw std::runtime_error(
        "CUDA INT8 expert execution requires native CUDA kernels");
#endif
}

void CudaTensorBackend::add(TensorView left,
                            TensorView right,
                            TensorView output) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    [[maybe_unused]] const auto leftOwner = pin(left, "CUDA add");
    [[maybe_unused]] const auto rightOwner = pin(right, "CUDA add");
    [[maybe_unused]] const auto outputOwner = pin(output, "CUDA add");
    validateCudaElementwise(left, right, output, impl_->ordinal, "CUDA add");
    if (left.data() == output.data() || right.data() == output.data()) {
        throw std::invalid_argument("CUDA add output cannot alias an input");
    }
    const auto elements = left.shape().elementCount();
    if (elements > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error("CUDA add exceeds cuBLAS integer limits");
    }
#ifdef HYPERMOE_HAS_CUBLAS
    const auto count = static_cast<int>(elements);
    const float one = 1.0F;
    auto operation = impl_->events->begin(profiling::GpuOperation::Elementwise,
        impl_->streams->stream(backend::CudaStreamRole::Compute), {leftOwner, rightOwner, outputOwner});
    checkCublas(cublasScopy(impl_->handle, count,
                            static_cast<const float*>(left.data()), 1,
                            static_cast<float*>(output.mutableData()), 1),
                "cublasScopy(add)");
    checkCublas(cublasSaxpy(impl_->handle, count, &one,
                            static_cast<const float*>(right.data()), 1,
                            static_cast<float*>(output.mutableData()), 1),
                "cublasSaxpy(add)");
    operation.finish();
#else
    (void)elements;
    throw std::runtime_error("cuBLAS support is unavailable");
#endif
}

void CudaTensorBackend::mul(TensorView left,
                            TensorView right,
                            TensorView output) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    [[maybe_unused]] const auto leftOwner = pin(left, "CUDA multiply");
    [[maybe_unused]] const auto rightOwner = pin(right, "CUDA multiply");
    [[maybe_unused]] const auto outputOwner = pin(output, "CUDA multiply");
    validateCudaElementwise(left, right, output, impl_->ordinal, "CUDA multiply");
    if (left.data() == output.data() || right.data() == output.data()) {
        throw std::invalid_argument("CUDA multiply output cannot alias an input");
    }
    const auto elements = left.shape().elementCount();
    if (elements > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error("CUDA multiply exceeds cuBLAS integer limits");
    }
#ifdef HYPERMOE_HAS_CUBLAS
    const auto count = static_cast<int>(elements);
    auto operation = impl_->events->begin(profiling::GpuOperation::Elementwise,
        impl_->streams->stream(backend::CudaStreamRole::Compute), {leftOwner, rightOwner, outputOwner});
    checkCublas(cublasSdgmm(
                    impl_->handle, CUBLAS_SIDE_RIGHT, 1, count,
                    static_cast<const float*>(left.data()), 1,
                    static_cast<const float*>(right.data()), 1,
                    static_cast<float*>(output.mutableData()), 1),
                "cublasSdgmm(multiply)");
    operation.finish();
#else
    (void)elements;
    throw std::runtime_error("cuBLAS support is unavailable");
#endif
}

void CudaTensorBackend::rmsNorm(TensorView input,
                                TensorView weight,
                                TensorView output,
                                float epsilon) {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    [[maybe_unused]] const auto inputOwner = pin(input, "CUDA RMSNorm");
    [[maybe_unused]] const auto weightOwner = pin(weight, "CUDA RMSNorm");
    [[maybe_unused]] const auto outputOwner = pin(output, "CUDA RMSNorm");
    if (!inputOwner || !weightOwner || !outputOwner || !input || !weight || !output ||
        input.device() != device() || weight.device() != device() ||
        output.device() != device() || input.dtype() != DType::FP32 ||
        weight.dtype() != DType::FP32 || output.dtype() != DType::FP32 ||
        !input.isContiguous() || !weight.isContiguous() || !output.isContiguous() ||
        !output.writable() || input.shape().rank() != 2 || weight.shape().rank() != 1 ||
        output.shape() != input.shape() ||
        weight.shape().dimensions()[0] != input.shape().dimensions()[1] ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument("CUDA RMSNorm tensor metadata is incompatible");
    }
    const auto rows = input.shape().dimensions()[0];
    const auto width = input.shape().dimensions()[1];
    if (width > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error("CUDA RMSNorm width exceeds cuBLAS integer limits");
    }
#ifdef HYPERMOE_HAS_CUBLAS
    auto operation = impl_->events->begin(profiling::GpuOperation::RMSNorm,
        impl_->streams->stream(backend::CudaStreamRole::Compute), {inputOwner, weightOwner, outputOwner});
    if (nativeKernelsAvailable()) {
#ifdef HYPERMOE_HAS_CUDA_KERNELS
        backend::cuda::kernels::rmsNorm(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(weight.data()),
            static_cast<float*>(output.mutableData()), rows, width, epsilon,
            impl_->streams->stream(backend::CudaStreamRole::Compute));
        operation.finish();
        return;
#endif
    }
    const auto count = static_cast<int>(width);
    const auto* source = static_cast<const float*>(input.data());
    const auto* scale = static_cast<const float*>(weight.data());
    auto* destination = static_cast<float*>(output.mutableData());
    for (std::size_t row = 0; row < rows; ++row) {
        float norm{};
        checkCublas(cublasSnrm2(impl_->handle, count, source + row * width, 1, &norm),
                    "cublasSnrm2(RMSNorm)");
        const auto inverse = 1.0F / std::sqrt(
            (norm * norm) / static_cast<float>(width) + epsilon);
        checkCublas(cublasSdgmm(
                        impl_->handle, CUBLAS_SIDE_RIGHT, 1, count,
                        source + row * width, 1, scale, 1,
                        destination + row * width, 1),
                    "cublasSdgmm(RMSNorm)");
        checkCublas(cublasSscal(impl_->handle, count, &inverse,
                                destination + row * width, 1),
                    "cublasSscal(RMSNorm)");
    }
    operation.finish();
#else
    (void)rows;
    (void)width;
    throw std::runtime_error("cuBLAS support is unavailable");
#endif
}

void CudaTensorBackend::applyActivation(int type,
                                        TensorView input,
                                        TensorView output) {
    validateCudaElementwise(input, input, output, impl_->ordinal, "CUDA activation");
    if (type < 0 || type > 1) {
        throw std::invalid_argument("CUDA activation type is invalid");
    }
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    auto operation = impl_->events->begin(profiling::GpuOperation::Activation,
        impl_->streams->stream(backend::CudaStreamRole::Compute),
        {pin(input, "CUDA activation"), pin(output, "CUDA activation")});
    backend::cuda::kernels::activation(
        type, static_cast<const float*>(input.data()),
        static_cast<float*>(output.mutableData()), input.shape().elementCount(),
        impl_->streams->stream(backend::CudaStreamRole::Compute));
    operation.finish();
#else
    (void)type;
    throw std::runtime_error("native CUDA activation kernels are unavailable");
#endif
}

void CudaTensorBackend::applyRoPE(TensorView values,
                                  std::size_t tokenCount,
                                  std::size_t headCount,
                                  std::size_t headDimension,
                                  std::size_t positionOffset,
                                  float theta) {
    [[maybe_unused]] const auto owner = pin(values, "CUDA RoPE");
    const auto expectedElements = checkedProduct(
        checkedProduct(tokenCount, headCount, "CUDA RoPE"), headDimension,
        "CUDA RoPE");
    if (!owner || !values || values.device() != device() ||
        values.dtype() != DType::FP32 || !values.isContiguous() ||
        !values.writable() || tokenCount == 0 || headCount == 0 ||
        headDimension == 0 || headDimension % 2U != 0 ||
        values.shape().elementCount() != expectedElements ||
        positionOffset > std::numeric_limits<std::size_t>::max() - tokenCount ||
        !std::isfinite(theta) || theta <= 0.0F) {
        throw std::invalid_argument("CUDA RoPE tensor metadata is invalid");
    }
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    auto operation = impl_->events->begin(profiling::GpuOperation::RoPE,
        impl_->streams->stream(backend::CudaStreamRole::Compute), {owner});
    backend::cuda::kernels::rope(
        static_cast<float*>(values.mutableData()), tokenCount, headCount,
        headDimension, positionOffset, theta,
        impl_->streams->stream(backend::CudaStreamRole::Compute));
    operation.finish();
#else
    (void)positionOffset;
    throw std::runtime_error("native CUDA RoPE kernels are unavailable");
#endif
}

CudaTensorBackend::RoutingSelection CudaTensorBackend::routeTopK(
    TensorView hiddenStates, TensorView routerWeights,
    std::size_t expertCount, std::size_t topK,
    bool softmax, bool renormalize) {
    if (!nativeKernelsAvailable() || !hiddenStates || !routerWeights ||
        hiddenStates.device() != device() || routerWeights.device() != device() ||
        hiddenStates.dtype() != DType::FP32 || routerWeights.dtype() != DType::FP32 ||
        !hiddenStates.isContiguous() || !routerWeights.isContiguous() ||
        hiddenStates.shape().rank() != 2 || routerWeights.shape().rank() != 2 ||
        expertCount == 0 || topK == 0 || topK > expertCount ||
        routerWeights.shape().dimensions()[0] != hiddenStates.shape().dimensions()[1] ||
        routerWeights.shape().dimensions()[1] != expertCount ||
        expertCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA router tensor metadata is incompatible");
    }
    const auto tokens = hiddenStates.shape().dimensions()[0];
    const auto selectedCount = checkedProduct(tokens, topK, "CUDA router");
    auto logits = allocateTensor({tokens, expertCount}, DType::FP32);
    matmul(hiddenStates, routerWeights, logits.view());
    auto idsBuffer = impl_->pool->allocateDeviceBuffer(
        selectedCount * sizeof(std::uint32_t));
    auto scoresBuffer = impl_->pool->allocateDeviceBuffer(selectedCount * sizeof(float));
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    const auto stream = impl_->streams->stream(backend::CudaStreamRole::Compute);
    auto operation = impl_->events->begin(profiling::GpuOperation::Router,
        stream, {pin(logits.view(), "CUDA router"), idsBuffer, scoresBuffer});
    backend::cuda::kernels::routerTopK(
        static_cast<float*>(logits.data()), tokens, expertCount, topK,
        softmax, renormalize,
        static_cast<std::uint32_t*>(idsBuffer->data()),
        static_cast<float*>(scoresBuffer->data()), stream);
    operation.finish();
    RoutingSelection result;
    result.expertIds.resize(selectedCount);
    result.scores.resize(selectedCount);
    impl_->backend->copyFromDevice(result.expertIds.data(), idsBuffer->data(),
                                   idsBuffer->size(), stream);
    impl_->backend->copyFromDevice(result.scores.data(), scoresBuffer->data(),
                                   scoresBuffer->size(), stream);
    impl_->backend->synchronize(stream);
    impl_->events->collect();
    if (std::any_of(result.scores.begin(), result.scores.end(),
                    [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("CUDA router produced non-finite scores");
    }
    return result;
#else
    (void)softmax;
    (void)renormalize;
    throw std::runtime_error("native CUDA router kernels are unavailable");
#endif
}

void CudaTensorBackend::causalAttention(
    TensorView query, TensorView key, TensorView value,
    TensorView scores, TensorView probabilities, TensorView context,
    std::size_t queryHeads, std::size_t keyValueHeads,
    std::size_t headDimension, std::uint64_t queryPositionOffset,
    std::uint64_t keyPositionOffset, bool causal) {
    const auto valid = [&](TensorView tensor) {
        return tensor && tensor.device() == device() &&
               tensor.dtype() == DType::FP32 && tensor.isContiguous();
    };
    const auto expectedQueryWidth = checkedProduct(
        queryHeads, headDimension, "CUDA attention query");
    if (!nativeKernelsAvailable() || !valid(query) || !valid(key) || !valid(value) ||
        !valid(scores) || !valid(probabilities) || !valid(context) ||
        !scores.writable() || !probabilities.writable() || !context.writable() ||
        query.shape().rank() != 2 || key.shape().rank() != 3 ||
        value.shape() != key.shape() || scores.shape().rank() != 3 ||
        probabilities.shape() != scores.shape() || context.shape() != query.shape() ||
        queryHeads == 0 || keyValueHeads == 0 || headDimension == 0 ||
        queryHeads % keyValueHeads != 0 ||
        query.shape().dimensions()[0] > std::numeric_limits<unsigned>::max() ||
        queryHeads > std::numeric_limits<unsigned>::max() ||
        query.shape().dimensions()[1] != expectedQueryWidth ||
        key.shape().dimensions()[1] != keyValueHeads ||
        key.shape().dimensions()[2] != headDimension ||
        scores.shape().dimensions() != std::vector<std::size_t>{
            queryHeads, query.shape().dimensions()[0], key.shape().dimensions()[0]} ||
        queryPositionOffset > std::numeric_limits<std::uint64_t>::max() -
            query.shape().dimensions()[0] ||
        keyPositionOffset > std::numeric_limits<std::uint64_t>::max() -
            key.shape().dimensions()[0] ||
        (causal && queryPositionOffset < keyPositionOffset)) {
        throw std::invalid_argument("CUDA attention tensor metadata is incompatible");
    }
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    auto operation = impl_->events->begin(profiling::GpuOperation::AttentionCore,
        impl_->streams->stream(backend::CudaStreamRole::Compute),
        {pin(query, "CUDA attention"), pin(key, "CUDA attention"), pin(value, "CUDA attention"),
         pin(scores, "CUDA attention"), pin(probabilities, "CUDA attention"), pin(context, "CUDA attention")});
    backend::cuda::kernels::causalAttention(
        static_cast<const float*>(query.data()), static_cast<const float*>(key.data()),
        static_cast<const float*>(value.data()), static_cast<float*>(scores.mutableData()),
        static_cast<float*>(probabilities.mutableData()),
        static_cast<float*>(context.mutableData()), query.shape().dimensions()[0],
        key.shape().dimensions()[0], queryHeads, keyValueHeads, headDimension,
        queryPositionOffset, keyPositionOffset, causal,
        impl_->streams->stream(backend::CudaStreamRole::Compute));
    operation.finish();
#else
    (void)queryPositionOffset;
    (void)keyPositionOffset;
    (void)causal;
    throw std::runtime_error("native CUDA attention kernels are unavailable");
#endif
}

void CudaTensorBackend::gatherRows(TensorView input,
                                   std::span<const std::size_t> rows,
                                   TensorView output) {
    if (!nativeKernelsAvailable() || !input || !output || rows.empty() ||
        input.device() != device() || output.device() != device() ||
        input.dtype() != DType::FP32 || output.dtype() != DType::FP32 ||
        !input.isContiguous() || !output.isContiguous() || !output.writable() ||
        input.shape().rank() != 2 || output.shape() != Shape{
            rows.size(), input.shape().dimensions()[1]}) {
        throw std::invalid_argument("CUDA gather tensor metadata is incompatible");
    }
    std::vector<std::uint32_t> indices;
    indices.reserve(rows.size());
    for (const auto row : rows) {
        if (row >= input.shape().dimensions()[0] ||
            row > std::numeric_limits<std::uint32_t>::max()) {
            throw std::out_of_range("CUDA gather row is invalid");
        }
        indices.push_back(static_cast<std::uint32_t>(row));
    }
    auto indexBuffer = impl_->pool->allocateDeviceBuffer(indices.size() * sizeof(std::uint32_t));
    const auto stream = impl_->streams->stream(backend::CudaStreamRole::Compute);
    impl_->backend->copyToDevice(indexBuffer->data(), indices.data(), indexBuffer->size(), stream);
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    auto operation = impl_->events->begin(profiling::GpuOperation::Gather, stream,
        {pin(input, "CUDA gather"), pin(output, "CUDA gather"), indexBuffer});
    backend::cuda::kernels::gatherRows(
        static_cast<const float*>(input.data()), static_cast<float*>(output.mutableData()),
        static_cast<const std::uint32_t*>(indexBuffer->data()), rows.size(),
        input.shape().dimensions()[1], stream);
    operation.finish();
    impl_->backend->synchronize(stream);
    impl_->events->collect();
#endif
}

void CudaTensorBackend::scatterAddRows(
    TensorView input, std::span<const std::size_t> rows,
    std::span<const float> weights, TensorView output) {
    if (!nativeKernelsAvailable() || !input || !output || rows.empty() ||
        rows.size() != weights.size() || input.device() != device() ||
        output.device() != device() || input.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32 || !input.isContiguous() ||
        !output.isContiguous() || !output.writable() || input.shape().rank() != 2 ||
        output.shape().rank() != 2 || input.shape().dimensions()[0] != rows.size() ||
        input.shape().dimensions()[1] != output.shape().dimensions()[1]) {
        throw std::invalid_argument("CUDA scatter tensor metadata is incompatible");
    }
    std::vector<std::uint32_t> indices;
    indices.reserve(rows.size());
    for (const auto row : rows) {
        if (row >= output.shape().dimensions()[0] ||
            row > std::numeric_limits<std::uint32_t>::max()) {
            throw std::out_of_range("CUDA scatter row is invalid");
        }
        indices.push_back(static_cast<std::uint32_t>(row));
    }
    auto indexBuffer = impl_->pool->allocateDeviceBuffer(indices.size() * sizeof(std::uint32_t));
    auto weightBuffer = impl_->pool->allocateDeviceBuffer(weights.size_bytes());
    const auto stream = impl_->streams->stream(backend::CudaStreamRole::Compute);
    impl_->backend->copyToDevice(indexBuffer->data(), indices.data(), indexBuffer->size(), stream);
    impl_->backend->copyToDevice(weightBuffer->data(), weights.data(), weights.size_bytes(), stream);
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    auto operation = impl_->events->begin(profiling::GpuOperation::Scatter, stream,
        {pin(input, "CUDA scatter"), pin(output, "CUDA scatter"), indexBuffer, weightBuffer});
    backend::cuda::kernels::scatterAddRows(
        static_cast<const float*>(input.data()), static_cast<float*>(output.mutableData()),
        static_cast<const std::uint32_t*>(indexBuffer->data()),
        static_cast<const float*>(weightBuffer->data()), rows.size(),
        input.shape().dimensions()[1], stream);
    operation.finish();
    impl_->backend->synchronize(stream);
    impl_->events->collect();
#endif
}

void CudaTensorBackend::zero(TensorView output) {
    if (!nativeKernelsAvailable() || !output || output.device() != device() ||
        output.dtype() != DType::FP32 || !output.isContiguous() || !output.writable()) {
        throw std::invalid_argument("CUDA zero tensor metadata is incompatible");
    }
#ifdef HYPERMOE_HAS_CUDA_KERNELS
    auto operation = impl_->events->begin(profiling::GpuOperation::Zero,
        impl_->streams->stream(backend::CudaStreamRole::Compute), {pin(output, "CUDA zero")});
    backend::cuda::kernels::zero(
        static_cast<float*>(output.mutableData()), output.shape().elementCount(),
        impl_->streams->stream(backend::CudaStreamRole::Compute));
    operation.finish();
#endif
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
    impl_->events->collect();
    if (impl_->profiler) impl_->profiler->recordSynchronization();
}

void CudaTensorBackend::synchronizeExecution() {
    if (!available()) throw std::runtime_error("CUDA tensor backend is unavailable");
    impl_->backend->synchronize(
        impl_->streams->stream(backend::CudaStreamRole::Compute));
    impl_->events->collect();
    if (impl_->profiler) impl_->profiler->recordSynchronization();
}

} // namespace hypermoe::tensor
