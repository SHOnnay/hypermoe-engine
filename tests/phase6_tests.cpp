#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/cache_policy.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/TensorView.hpp"
#include "tensor/activation/Activation.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "tensor/quantization/QuantizedTensor.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
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
    expect(tensor.shape().elementCount() == values.size(),
           "test data matches tensor shape");
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.data()));
}

bool near(const hypermoe::tensor::Tensor& tensor,
          std::initializer_list<float> expected,
          float tolerance = 1.0e-5F) {
    if (tensor.shape().elementCount() != expected.size()) return false;
    const auto* values = static_cast<const float*>(tensor.data());
    std::size_t index = 0;
    for (const auto value : expected) {
        if (std::fabs(values[index++] - value) > tolerance) return false;
    }
    return true;
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-phase6-" +
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

void testTensorView() {
    using namespace hypermoe::tensor;
    CpuTensorBackend backend;
    auto tensor = backend.allocateTensor({2, 4}, DType::FP32);
    auto view = tensor.view();
    expect(view.valid() && view.writable() && view.data() == tensor.data(),
           "TensorView references owned tensor storage without copying");
    auto slice = view.sliceBytes(4 * sizeof(float), Shape{1, 4}, DType::FP32);
    expect(slice.data() == static_cast<std::byte*>(tensor.data()) + 4 * sizeof(float) &&
               slice.bytes() == 4 * sizeof(float),
           "TensorView creates checked byte-offset subviews");
    expectThrows([&] { (void)view.sliceBytes(1, Shape{1}, DType::FP32); },
                 "TensorView rejects misaligned slices");
    expectThrows([&] { (void)view.sliceBytes(32, Shape{1}, DType::FP32); },
                 "TensorView rejects out-of-range slices");
    expectThrows([&] { (void)slice.sliceBytes(16, Shape{1}, DType::FP32); },
                 "TensorView slices cannot escape their declared span");

    const Tensor& readOnlyTensor = tensor;
    const TensorView readOnly(readOnlyTensor);
    expect(!readOnly.writable() && readOnly.mutableData() == nullptr,
           "const tensors produce read-only views");

    TensorView expired;
    {
        auto owner = std::make_shared<std::vector<std::byte>>(16);
        auto temporary = Tensor::fromStorage(
            Shape{4}, DType::FP32, Device::cpu(), owner->data(), owner->size(),
            std::static_pointer_cast<void>(owner));
        expired = temporary.view();
    }
    expect(!expired.valid() && expired.data() == nullptr,
           "TensorView detects expiration without extending ownership");
}

void testQuantizedTensor() {
    using namespace hypermoe::tensor;
    using namespace hypermoe::tensor::quantization;
    expect(storageSizeBytes(Shape{5}, QuantizedDType::INT8) == 5,
           "INT8 storage uses one byte per element");
    expect(storageSizeBytes(Shape{5}, QuantizedDType::Q4) == 3,
           "Q4 storage packs two values per byte with odd-element rounding");
    expectThrows(
        [] { (void)storageSizeBytes(Shape({2, 2}, {3, 1}), QuantizedDType::Q4); },
        "packed quantization rejects strided layouts");
    expectThrows(
        [] { validateParameters(QuantizedDType::INT8, {0.0F, 0}); },
        "quantization rejects nonpositive scale");
    expectThrows(
        [] { validateParameters(QuantizedDType::Q4, {0.25F, 8}); },
        "Q4 quantization rejects an out-of-range zero point");

    auto bytes = std::make_shared<std::vector<std::byte>>(3);
    auto quantized = QuantizedTensor::fromStorage(
        Shape{5}, QuantizedDType::Q4, {0.125F, -1}, Device::cpu(), bytes->data(),
        bytes->size(), std::static_pointer_cast<void>(bytes));
    expect(quantized.valid() && quantized.bytes() == 3 &&
               quantized.parameters().scale == 0.125F,
           "QuantizedTensor validates and owns packed storage");
    const auto json = quantized.metadata().toJson();
    expect(json.find("\"dtype\":\"Q4\"") != std::string::npos &&
               json.find("\"shape\":[5]") != std::string::npos &&
               json.find("\"storage_bytes\":3") != std::string::npos,
           "quantization metadata serializes dtype, shape, and storage size");
    auto undersized = std::make_shared<std::vector<std::byte>>(2);
    expectThrows(
        [&] {
            (void)QuantizedTensor::fromStorage(
                Shape{5}, QuantizedDType::Q4, {0.25F, 0}, Device::cpu(),
                undersized->data(), undersized->size(),
                std::static_pointer_cast<void>(undersized));
        },
        "QuantizedTensor rejects undersized packed storage");
}

void testActivations() {
    using namespace hypermoe::tensor;
    CpuTensorBackend backend;
    auto input = backend.allocateTensor({5}, DType::FP32);
    auto output = backend.allocateTensor({5}, DType::FP32);
    writeFloats(input, {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    activation::apply(activation::ActivationType::SiLU, backend, input, output);
    expect(near(output, {-0.23840584F, -0.26894143F, 0.0F, 0.73105860F,
                         1.76159418F}),
           "SiLU matches reference values");
    activation::apply(activation::ActivationType::GELU, backend, input, output);
    expect(near(output, {-0.04550026F, -0.15865526F, 0.0F, 0.84134471F,
                         1.95449972F}),
           "exact GELU matches reference values");

    const Tensor& constOutput = output;
    expectThrows(
        [&] {
            activation::apply(activation::ActivationType::SiLU, backend,
                              input.view(), constOutput.view());
        },
        "activation rejects a read-only output view");
    expectThrows(
        [&] {
            activation::apply(static_cast<activation::ActivationType>(99), backend,
                              input, output);
        },
        "activation rejects unknown enum values");
}

void testExpertMlp() {
    using namespace hypermoe::tensor;
    auto profiler = std::make_shared<hypermoe::Profiler>();
    auto backend = std::make_shared<CpuTensorBackend>(profiler);
    auto input = backend->allocateTensor({1, 2}, DType::FP32);
    auto gate = backend->allocateTensor({2, 2}, DType::FP32);
    auto up = backend->allocateTensor({2, 2}, DType::FP32);
    auto down = backend->allocateTensor({2, 1}, DType::FP32);
    auto output = backend->allocateTensor({1, 1}, DType::FP32);
    writeFloats(input, {1.0F, 2.0F});
    writeFloats(gate, {1.0F, 0.0F, 0.0F, 1.0F});
    writeFloats(up, {2.0F, 0.0F, 0.0F, 3.0F});
    writeFloats(down, {1.0F, 0.5F});

    hypermoe::ExpertMlpExecutor executor(backend,
        activation::ActivationType::SiLU, profiler);
    executor.execute(input, {gate, up, down}, output);
    const float expected = activation::silu(1.0F) * 2.0F +
                           activation::silu(2.0F) * 6.0F * 0.5F;
    expect(near(output, {expected}),
           "gated expert MLP executes gate, activation, up, and down projections");
    const auto metrics = profiler->snapshot();
    expect(metrics.expertExecutionTime.count() > 0 &&
               metrics.activationTime.count() > 0 &&
               metrics.projectionTime.count() > 0,
           "expert execution profiling records total, activation, and projections");
    profiler->recordQuantizationTime(std::chrono::nanoseconds(17));
    expect(profiler->toJson().find("\"quantization_time_ms\"") !=
               std::string::npos,
           "profiler exports all Phase 6 timing fields");

    auto invalidDown = backend->allocateTensor({3, 1}, DType::FP32);
    expectThrows([&] { executor.execute(input, {gate, up, invalidDown}, output); },
                 "expert MLP rejects incompatible projection dimensions");
}

void testExpertManagerViewsAndQuantization() {
    using namespace hypermoe::tensor;
    TemporaryDirectory directory;
    const std::vector<float> gate{1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> up{2.0F, 0.0F, 0.0F, 3.0F};
    const std::vector<float> down{1.0F, 0.0F, 0.0F, 1.0F};
    std::vector<std::byte> packed((gate.size() + up.size() + down.size()) *
                                  sizeof(float));
    std::size_t offset = 0;
    for (const auto* values : {&gate, &up, &down}) {
        const auto bytes = values->size() * sizeof(float);
        std::memcpy(packed.data() + offset, values->data(), bytes);
        offset += bytes;
    }
    const std::vector<std::byte> q4{std::byte{0x12}, std::byte{0x34},
                                    std::byte{0x05}};
    const std::vector<hypermoe::storage::ExpertBlob> blobs{
        {0, 7, static_cast<std::uint32_t>(hypermoe::QuantizationType::Fp32), packed},
        {0, 8, static_cast<std::uint32_t>(hypermoe::QuantizationType::Q4), q4},
    };
    hypermoe::storage::ExpertStore::create(directory.path(), blobs, "{\"phase\":6}");
    auto store = std::make_shared<hypermoe::storage::ExpertStore>(directory.path());
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
    auto compute = std::make_shared<hypermoe::backend::CpuBackend>();
    auto transfers = std::make_shared<hypermoe::TransferManager>(loader, compute, 1);
    hypermoe::MemoryManager memory(4096, 4096);
    hypermoe::ExpertManager manager(
        memory, std::make_unique<hypermoe::LruCachePolicy>(), transfers);
    manager.registerExpert({7, 0, packed.size(), hypermoe::QuantizationType::Fp32,
                            hypermoe::MemoryTier::Nvme});
    manager.registerExpert({8, 0, q4.size(), hypermoe::QuantizationType::Q4,
                            hypermoe::MemoryTier::Nvme});
    (void)manager.requestExpert(7);
    (void)manager.requestExpert(8);

    const auto buffer = manager.residentDeviceWeights(7);
    const auto ownersBefore = buffer.use_count();
    const auto raw = manager.residentDeviceTensorView(
        7, Shape{packed.size()}, DType::INT8);
    expect(raw.data() == buffer->data() && buffer.use_count() == ownersBefore,
           "ExpertManager returns a non-owning zero-copy view of its DeviceBuffer");
    const auto matrixBytes = gate.size() * sizeof(float);
    const hypermoe::ExpertMlpWeights weights{
        raw.sliceBytes(0, Shape{2, 2}, DType::FP32),
        raw.sliceBytes(matrixBytes, Shape{2, 2}, DType::FP32),
        raw.sliceBytes(2 * matrixBytes, Shape{2, 2}, DType::FP32),
    };
    auto tensorBackend = std::make_shared<CpuTensorBackend>();
    auto input = tensorBackend->allocateTensor({1, 2}, DType::FP32);
    auto output = tensorBackend->allocateTensor({1, 2}, DType::FP32);
    writeFloats(input, {1.0F, 2.0F});
    hypermoe::ExpertMlpExecutor executor(tensorBackend);
    executor.execute(input, weights, output);
    expect(near(output, {activation::silu(1.0F) * 2.0F,
                         activation::silu(2.0F) * 6.0F}),
           "resident expert slices execute without duplicate weight allocation");

    const auto quantized = manager.residentQuantizedTensor(
        8, Shape{5}, quantization::QuantizedDType::Q4, {0.25F, 0});
    expect(quantized.data() == manager.residentDeviceWeights(8)->data() &&
               quantized.bytes() == 3,
           "ExpertManager creates an owning Q4 descriptor over resident bytes");
    expectThrows(
        [&] {
            (void)manager.residentQuantizedTensor(
                8, Shape{3}, quantization::QuantizedDType::INT8, {0.25F, 0});
        },
        "ExpertManager rejects quantization metadata mismatches");
}

void testCudaExpertMlp() {
    using namespace hypermoe::tensor;
    auto cuda = std::make_shared<CudaTensorBackend>();
    if (!cuda->available()) {
        std::cout << "Phase 6 CUDA expert test skipped: cuBLAS runtime/device unavailable\n";
        return;
    }
    CpuTensorBackend cpu;
    auto hostInput = cpu.allocateTensor({1, 2}, DType::FP32);
    auto hostGate = cpu.allocateTensor({2, 2}, DType::FP32);
    auto hostUp = cpu.allocateTensor({2, 2}, DType::FP32);
    auto hostDown = cpu.allocateTensor({2, 1}, DType::FP32);
    auto expected = cpu.allocateTensor({1, 1}, DType::FP32);
    auto actual = cpu.allocateTensor({1, 1}, DType::FP32);
    writeFloats(hostInput, {1.0F, 2.0F});
    writeFloats(hostGate, {1.0F, 0.0F, 0.0F, 1.0F});
    writeFloats(hostUp, {2.0F, 0.0F, 0.0F, 3.0F});
    writeFloats(hostDown, {1.0F, 0.5F});
    auto cpuBackend = std::make_shared<CpuTensorBackend>();
    hypermoe::ExpertMlpExecutor cpuExecutor(cpuBackend);
    cpuExecutor.execute(hostInput, {hostGate, hostUp, hostDown}, expected);

    auto deviceInput = cuda->allocateTensor(hostInput.shape(), DType::FP32);
    auto deviceGate = cuda->allocateTensor(hostGate.shape(), DType::FP32);
    auto deviceUp = cuda->allocateTensor(hostUp.shape(), DType::FP32);
    auto deviceDown = cuda->allocateTensor(hostDown.shape(), DType::FP32);
    auto deviceOutput = cuda->allocateTensor(expected.shape(), DType::FP32);
    cuda->copyTensor(hostInput, deviceInput);
    cuda->copyTensor(hostGate, deviceGate);
    cuda->copyTensor(hostUp, deviceUp);
    cuda->copyTensor(hostDown, deviceDown);
    hypermoe::ExpertMlpExecutor cudaExecutor(cuda);
    cudaExecutor.execute(deviceInput, {deviceGate, deviceUp, deviceDown}, deviceOutput);
    cuda->copyTensor(deviceOutput, actual);
    expect(near(actual, {static_cast<const float*>(expected.data())[0]}),
           "CUDA expert MLP matches CPU reference within FP32 tolerance");
}

} // namespace

int main() {
    testTensorView();
    testQuantizedTensor();
    testActivations();
    testExpertMlp();
    testExpertManagerViewsAndQuantization();
    testCudaExpertMlp();
    if (failures != 0) {
        std::cerr << failures << " Phase 6 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 6 tests passed\n";
    return EXIT_SUCCESS;
}
