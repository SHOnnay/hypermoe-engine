#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/cache_policy.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void expectThrows(Function&& function, std::string_view message) {
    try {
        function();
        expect(false, message);
    } catch (const std::exception&) {
        expect(true, message);
    }
}

void writeFloats(hypermoe::tensor::Tensor& tensor,
                 std::initializer_list<float> values) {
    expect(tensor.dtype() == hypermoe::tensor::DType::FP32 &&
               tensor.shape().elementCount() == values.size(),
           "test float data matches tensor metadata");
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.data()));
}

bool near(const hypermoe::tensor::Tensor& tensor,
          std::initializer_list<float> expected,
          float tolerance = 1.0e-5F) {
    if (tensor.dtype() != hypermoe::tensor::DType::FP32 ||
        tensor.shape().elementCount() != expected.size()) {
        return false;
    }
    const auto* values = static_cast<const float*>(tensor.data());
    std::size_t index = 0;
    for (const auto expectedValue : expected) {
        if (std::fabs(values[index++] - expectedValue) > tolerance) return false;
    }
    return true;
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-phase5-" +
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

void testShapeAndTensorValidation() {
    using hypermoe::tensor::DType;
    using hypermoe::tensor::Shape;
    const Shape contiguous{2, 3, 4};
    expect(contiguous.rank() == 3 && contiguous.elementCount() == 24 &&
               contiguous.storageElementCount() == 24 &&
               contiguous.strides() == std::vector<std::size_t>({12, 4, 1}) &&
               contiguous.isContiguous(),
           "shape creates checked contiguous row-major strides");
    const Shape strided({2, 2}, {3, 1});
    expect(strided.elementCount() == 4 && strided.storageElementCount() == 5 &&
               !strided.isContiguous(),
           "shape accounts for a non-contiguous storage span");
    expectThrows([] { (void)Shape{2, 0}; }, "zero tensor dimension is rejected");
    expectThrows(
        [] {
            (void)Shape{std::vector<std::size_t>{2, 2},
                        std::vector<std::size_t>{1}};
        },
        "stride rank mismatch is rejected");
    expectThrows(
        [] {
            (void)Shape{std::numeric_limits<std::size_t>::max(), 2};
        },
        "shape element overflow is rejected");

    hypermoe::tensor::CpuTensorBackend backend;
    auto stridedTensor = backend.allocateTensor(strided, DType::FP32);
    expect(stridedTensor.bytes() == 16 && stridedTensor.storageBytes() == 20,
           "tensor distinguishes logical bytes from a safe strided storage span");
    auto fp16 = backend.allocateTensor(Shape{7}, DType::FP16);
    auto int8 = backend.allocateTensor(Shape{7}, DType::INT8);
    expect(fp16.bytes() == 14 && int8.bytes() == 7,
           "tensor allocation supports FP16 and INT8 storage metadata");
    auto fp32 = backend.allocateTensor(Shape{2, 3}, DType::FP32);
    auto reshaped = backend.reshape(fp32, Shape{3, 2});
    expect(reshaped.data() == fp32.data() && reshaped.shape() == Shape({3, 2}),
           "reshape is a zero-copy shared-storage view");
    expectThrows([&] { (void)backend.reshape(fp32, Shape{7}); },
                 "reshape must preserve element count");
    auto undersized = std::make_shared<std::vector<std::byte>>(4);
    expectThrows(
        [&] {
            (void)hypermoe::tensor::Tensor::fromStorage(
                Shape{2}, DType::FP32, hypermoe::tensor::Device::cpu(),
                undersized->data(), undersized->size(),
                std::static_pointer_cast<void>(undersized));
        },
        "tensor rejects storage smaller than its metadata span");
}

void testCpuOperationsAndExpertExecutor() {
    auto profiler = std::make_shared<hypermoe::Profiler>();
    auto backend =
        std::make_shared<hypermoe::tensor::CpuTensorBackend>(profiler);
    auto left = backend->allocateTensor({2, 3}, hypermoe::tensor::DType::FP32);
    auto right = backend->allocateTensor({3, 2}, hypermoe::tensor::DType::FP32);
    auto output = backend->allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
    writeFloats(left, {1, 2, 3, 4, 5, 6});
    writeFloats(right, {7, 8, 9, 10, 11, 12});

    hypermoe::MatmulExpertExecutor executor(backend);
    executor.execute(left, right, output);
    expect(near(output, {58, 64, 139, 154}),
           "reference FP32 GEMM and expert executor produce expected output");

    auto copied = backend->allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
    backend->copyTensor(output, copied);
    expect(near(copied, {58, 64, 139, 154}), "CPU tensor copy preserves values");
    auto added = backend->allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
    auto multiplied = backend->allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
    backend->add(output, copied, added);
    backend->mul(output, copied, multiplied);
    expect(near(added, {116, 128, 278, 308}), "CPU elementwise add works");
    expect(near(multiplied, {3364, 4096, 19321, 23716}),
           "CPU elementwise multiply works");
    expectThrows(
        [&] {
            auto invalid = backend->allocateTensor(
                {3, 3}, hypermoe::tensor::DType::FP32);
            backend->matmul(left, right, invalid);
        },
        "matmul rejects an invalid output shape");

    profiler->observeGpuUtilization(47.5);
    const auto metrics = profiler->snapshot();
    expect(metrics.tensorAllocations == 7 && metrics.matmulTime.count() > 0 &&
               metrics.gpuUtilizationPercent == 47.5,
           "profiler tracks tensor allocations, matmul time, and utilization hook");
    const auto json = profiler->toJson();
    expect(json.find("\"kernel_time_ms\"") != std::string::npos &&
               json.find("\"tensor_allocations\": 7") != std::string::npos,
           "profiler exports Phase 5 metrics");
}

void testExpertManagerTensorBridge() {
    TemporaryDirectory directory;
    const std::vector<float> weights{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<std::byte> bytes(weights.size() * sizeof(float));
    std::memcpy(bytes.data(), weights.data(), bytes.size());
    const std::vector<hypermoe::storage::ExpertBlob> blobs{
        {0, 42, 0, bytes},
    };
    hypermoe::storage::ExpertStore::create(
        directory.path(), blobs, "{\"phase\":5}");
    auto store = std::make_shared<hypermoe::storage::ExpertStore>(directory.path());
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
    auto compute = std::make_shared<hypermoe::backend::CpuBackend>();
    auto transfers =
        std::make_shared<hypermoe::TransferManager>(loader, compute, 1);
    hypermoe::MemoryManager memory(4096, 4096);
    hypermoe::ExpertManager experts(
        memory, std::make_unique<hypermoe::LruCachePolicy>(), transfers);
    experts.registerExpert({42, 0, bytes.size(), hypermoe::QuantizationType::Fp32,
                            hypermoe::MemoryTier::Nvme});
    const auto requested = experts.requestExpert(42);
    expect(requested.expert.location == hypermoe::MemoryTier::Vram,
           "expert request moves real stored weights into a backend buffer");
    const auto tensor = experts.residentDeviceTensor(
        42, {2, 2}, hypermoe::tensor::DType::FP32);
    expect(tensor.device() == hypermoe::tensor::Device::cpu() &&
               near(tensor, {1, 2, 3, 4}),
           "ExpertManager exposes resident weights as a validated tensor view");
    expectThrows(
        [&] {
            (void)experts.residentDeviceTensor(
                42, {4, 2}, hypermoe::tensor::DType::FP32);
        },
        "expert tensor bridge rejects metadata that exceeds weight storage");
}

void testCudaBackendSwitching() {
    hypermoe::tensor::CpuTensorBackend cpu;
    hypermoe::tensor::CudaTensorBackend cuda;
    if (!cuda.available()) {
        expectThrows(
            [&] {
                (void)cuda.allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
            },
            "unavailable CUDA tensor backend fails cleanly");
        std::cout << "Phase 5 CUDA GEMM skipped: cuBLAS runtime/device unavailable\n";
        return;
    }

    auto hostLeft = cpu.allocateTensor({2, 3}, hypermoe::tensor::DType::FP32);
    auto hostRight = cpu.allocateTensor({3, 2}, hypermoe::tensor::DType::FP32);
    auto hostOutput = cpu.allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
    auto reference = cpu.allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
    writeFloats(hostLeft, {1, 2, 3, 4, 5, 6});
    writeFloats(hostRight, {7, 8, 9, 10, 11, 12});
    cpu.matmul(hostLeft, hostRight, reference);

    auto deviceLeft = cuda.allocateTensor({2, 3}, hypermoe::tensor::DType::FP32);
    auto deviceRight = cuda.allocateTensor({3, 2}, hypermoe::tensor::DType::FP32);
    auto deviceOutput = cuda.allocateTensor({2, 2}, hypermoe::tensor::DType::FP32);
    cuda.copyTensor(hostLeft, deviceLeft);
    cuda.copyTensor(hostRight, deviceRight);
    cuda.matmul(deviceLeft, deviceRight, deviceOutput);
    cuda.copyTensor(deviceOutput, hostOutput);
    const auto* actual = static_cast<const float*>(hostOutput.data());
    const auto* expected = static_cast<const float*>(reference.data());
    bool matches = true;
    for (std::size_t index = 0; index < 4; ++index) {
        matches = matches && std::fabs(actual[index] - expected[index]) <= 1.0e-5F;
    }
    expect(matches, "cuBLAS FP32 GEMM matches the CPU reference within 1e-5");
}

} // namespace

int main() {
    testShapeAndTensorValidation();
    testCpuOperationsAndExpertExecutor();
    testExpertManagerTensorBridge();
    testCudaBackendSwitching();
    if (failures != 0) {
        std::cerr << failures << " Phase 5 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 5 tests passed\n";
    return EXIT_SUCCESS;
}
