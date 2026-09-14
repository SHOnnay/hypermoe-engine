#include "backend/CpuBackend.hpp"
#include "cache/LRUPolicy.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "tensor/quantization/Quantization.hpp"
#include "tools/model_convert/ExpertPacker.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures{};

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

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
            ("hypermoe-phase22-" +
             std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

std::vector<std::byte> floatBytes(std::initializer_list<float> values) {
    std::vector<std::byte> result(values.size() * sizeof(float));
    std::memcpy(result.data(), values.begin(), result.size());
    return result;
}

void writeSafeTensor(const std::filesystem::path& path,
                     std::string header,
                     std::span<const std::byte> payload) {
    while (header.size() % 8U != 0U) header.push_back(' ');
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((size >> shift) & 0xffU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    if (!output) throw std::runtime_error("failed writing Phase 22 fixture");
}

void writeQwenExpertFixture(const std::filesystem::path& root) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Phase22-INT8","num_hidden_layers":1,"num_experts":2,"hidden_size":2,"moe_intermediate_size":2,"num_experts_per_tok":1,"norm_topk_prob":true})";
    const auto gateUp = floatBytes({
        1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F,
        0.5F, 0.0F, 0.0F, 0.5F, 1.0F, 0.0F, 0.0F, 1.0F});
    writeSafeTensor(
        root / "model-00001-of-00002.safetensors",
        R"({"model.layers.0.mlp.experts.gate_up_proj":{"dtype":"F32","shape":[2,4,2],"data_offsets":[0,64]},"__metadata__":{"format":"pt"}})",
        gateUp);
    const auto downRouter = floatBytes({
        1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F,
        3.0F, 0.0F, 0.0F, 1.0F});
    writeSafeTensor(
        root / "model-00002-of-00002.safetensors",
        R"({"model.layers.0.mlp.experts.down_proj":{"dtype":"F32","shape":[2,2,2],"data_offsets":[0,32]},"model.layers.0.mlp.gate.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[32,48]},"__metadata__":{"format":"pt"}})",
        downRouter);
    std::ofstream(root / "model.safetensors.index.json")
        << R"({"metadata":{"total_size":112},"weight_map":{"model.layers.0.mlp.experts.gate_up_proj":"model-00001-of-00002.safetensors","model.layers.0.mlp.experts.down_proj":"model-00002-of-00002.safetensors","model.layers.0.mlp.gate.weight":"model-00002-of-00002.safetensors"}})";
}

hypermoe::models::ExpertWeightMap makeMappings(
    const hypermoe::models::ModelManifest& manifest) {
    hypermoe::models::ExpertWeightMap result;
    for (const auto& expert : manifest.experts) {
        const auto add = [&](hypermoe::models::ExpertWeightType type,
                             const hypermoe::models::ProjectionLocation& projection) {
            const auto* metadata = manifest.findTensor(projection.tensorName);
            if (!metadata) throw std::logic_error("fixture projection is missing");
            result.add(
                expert.layerId, expert.expertId, type,
                {metadata->name, projection.shape, metadata->dtype,
                 metadata->quantization
                     ? std::optional{
                           hypermoe::tensor::quantization::QuantizedDType::INT8}
                     : std::nullopt,
                 projection.offset, projection.size, expert.layerId,
                 expert.expertId, metadata->quantization});
        };
        add(hypermoe::models::ExpertWeightType::GATE, expert.gate);
        add(hypermoe::models::ExpertWeightType::UP, expert.up);
        add(hypermoe::models::ExpertWeightType::DOWN, expert.down);
    }
    return result;
}

std::vector<float> executeExpert(const std::filesystem::path& artifact) {
    using namespace hypermoe;
    const auto manifest = models::ModelManifest::load(artifact / "manifest.json");
    storage::ExpertStore store(artifact);
    const auto record = store.index().find(0, 0);
    if (!record) throw std::logic_error("fixture expert is missing");
    auto payloadBytes = std::make_shared<const std::vector<std::byte>>(
        store.readExpert(0, 0, true));
    const auto payload = tensor::TensorView::fromHostBuffer(
        {payloadBytes->size()}, tensor::DType::INT8, payloadBytes, false);
    const auto weights = makeMappings(manifest).createViews(
        0, 0, payload, record->offset);
    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    auto input = backend->allocateTensor({1, 2}, tensor::DType::FP32);
    auto output = backend->allocateTensor({1, 2}, tensor::DType::FP32);
    const std::vector<float> values{1.0F, 2.0F};
    std::memcpy(input.data(), values.data(), values.size() * sizeof(float));
    ExpertMlpExecutor(backend).execute(input.view(), weights, output.view());
    const auto* data = static_cast<const float*>(output.data());
    return {data, data + output.shape().elementCount()};
}

void testQuantizerAndCpuReference() {
    using namespace hypermoe::tensor;
    using namespace hypermoe::tensor::quantization;
    const std::vector<float> values{-2.0F, -0.25F, 0.0F, 0.75F, 2.0F};
    const auto first = quantizeInt8(values);
    const auto second = quantizeInt8(values);
    const auto reconstructed = dequantizeInt8(first.bytes, first.parameters);
    expect(first.bytes == second.bytes && first.parameters == second.parameters,
           "INT8 quantization is deterministic");
    expect(first.parameters.zeroPoint == 0 && first.parameters.scale > 0.0F &&
               first.maximumAbsoluteError <= first.parameters.scale * 0.5001F,
           "symmetric INT8 scale bounds per-weight reconstruction error");
    expect(reconstructed.size() == values.size(),
           "INT8 storage dequantizes using persisted affine metadata");
    expectThrows([] { (void)quantizeInt8(std::vector<float>{}); },
                 "INT8 quantization rejects empty tensors");

    CpuTensorBackend backend;
    auto left = backend.allocateTensor({1, 3}, DType::FP32);
    auto fpWeights = backend.allocateTensor({3, 2}, DType::FP32);
    auto intWeights = backend.allocateTensor({3, 2}, DType::INT8);
    auto reference = backend.allocateTensor({1, 2}, DType::FP32);
    auto quantized = backend.allocateTensor({1, 2}, DType::FP32);
    const std::vector<float> leftValues{1.0F, -0.5F, 2.0F};
    const std::vector<float> weightValues{0.25F, -0.75F, 1.0F,
                                          0.5F, -0.5F, 0.125F};
    std::memcpy(left.data(), leftValues.data(), leftValues.size() * sizeof(float));
    std::memcpy(fpWeights.data(), weightValues.data(),
                weightValues.size() * sizeof(float));
    const auto packed = quantizeInt8(weightValues);
    std::memcpy(intWeights.data(), packed.bytes.data(), packed.bytes.size());
    backend.matmul(left.view(), fpWeights.view(), reference.view());
    backend.matmulInt8Weights(left.view(), intWeights.view(), packed.parameters,
                              quantized.view());
    const auto* expected = static_cast<const float*>(reference.data());
    const auto* actual = static_cast<const float*>(quantized.data());
    expect(std::fabs(expected[0] - actual[0]) < 0.02F &&
               std::fabs(expected[1] - actual[1]) < 0.02F,
           "CPU INT8 weight GEMM matches the FP32 reference within quantization error");
}

void testArtifactAndResidency() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source";
    const auto fp32 = temporary.path() / "fp32";
    const auto int8 = temporary.path() / "int8";
    std::filesystem::create_directories(source);
    writeQwenExpertFixture(source);
    const auto sourceManifest = importer::qwen::QwenImporter{}.inspect(source);
    const auto fp32Report = conversion::ExpertPacker{}.pack(
        sourceManifest, source, fp32);
    const auto int8Report = conversion::ExpertPacker{}.pack(
        sourceManifest, source, int8,
        {conversion::ExpertStorageEncoding::Int8});
    expect(fp32Report.validationPassed && int8Report.validationPassed &&
               int8Report.quantizedProjections == 6 &&
               int8Report.sourceExpertBytes == 4 * int8Report.packedExpertBytes,
           "expert packer emits validated 4x-smaller INT8 projection payloads");

    const auto manifest = models::ModelManifest::load(int8 / "manifest.json");
    const auto* gate = manifest.findTensor(manifest.experts.front().gate.tensorName);
    expect(gate && gate->dtype == tensor::DType::INT8 && gate->quantization &&
               gate->quantization->scale > 0.0F &&
               manifest.config.capabilities.quantizedExpertWeights,
           "manifest round-trips INT8 scale metadata and quantized capability");
    auto invalid = manifest;
    invalid.tensors.front().quantization.reset();
    expectThrows([&] { invalid.validate(); },
                 "manifest rejects INT8 tensors without scale metadata");

    auto store = std::make_shared<storage::ExpertStore>(int8);
    const auto record = store->index().find(0, 0);
    expect(record && record->quantization_type ==
                         static_cast<std::uint32_t>(QuantizationType::Int8),
           "expert index identifies the INT8 resident payload");
    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto compute = std::make_shared<backend::CpuBackend>();
    auto transfers = std::make_shared<TransferManager>(loader, compute, 1);
    MemoryManager memory(4096, 4096);
    ExpertManager manager(memory, std::make_unique<LruCachePolicy>(), transfers);
    manager.registerExpert({0, 0, static_cast<std::size_t>(record->size),
                            QuantizationType::Int8, MemoryTier::Nvme});
    {
        auto lease = manager.acquireHostExpert(0, 0);
        const auto view = lease.view({static_cast<std::size_t>(record->size)},
                                     tensor::DType::INT8);
        const auto resident = manager.findExpert(0, 0);
        expect(view.bytes() == record->size && resident &&
                   resident->location == MemoryTier::Ram,
               "INT8 expert remains compressed through NVMe-to-RAM residency");
    }

    const auto fp32Output = executeExpert(fp32);
    const auto int8Output = executeExpert(int8);
    expect(fp32Output.size() == int8Output.size() &&
               std::fabs(fp32Output[0] - int8Output[0]) < 0.03F &&
               std::fabs(fp32Output[1] - int8Output[1]) < 0.03F,
           "packed INT8 expert execution matches packed FP32 expert output");
}

void testCudaReferenceWhenAvailable() {
    using namespace hypermoe::tensor;
    CudaTensorBackend cuda;
    if (!cuda.available() || !cuda.nativeKernelsAvailable()) {
        std::cout << "CUDA INT8 test skipped: native CUDA kernels unavailable\n";
        return;
    }
    CpuTensorBackend cpu;
    auto hostInput = cpu.allocateTensor({1, 2}, DType::FP32);
    auto hostWeights = cpu.allocateTensor({2, 2}, DType::INT8);
    auto hostOutput = cpu.allocateTensor({1, 2}, DType::FP32);
    const std::vector<float> inputValues{2.0F, -1.0F};
    const std::vector<float> weightValues{0.5F, 1.0F, -0.25F, 0.75F};
    const auto packed = quantization::quantizeInt8(weightValues);
    std::memcpy(hostInput.data(), inputValues.data(), inputValues.size() * sizeof(float));
    std::memcpy(hostWeights.data(), packed.bytes.data(), packed.bytes.size());
    auto deviceInput = cuda.allocateTensor({1, 2}, DType::FP32);
    auto deviceWeights = cuda.allocateTensor({2, 2}, DType::INT8);
    auto deviceOutput = cuda.allocateTensor({1, 2}, DType::FP32);
    cuda.copyTensor(hostInput.view(), deviceInput.view());
    cuda.copyTensor(hostWeights.view(), deviceWeights.view());
    cuda.matmulInt8Weights(deviceInput.view(), deviceWeights.view(),
                           packed.parameters, deviceOutput.view());
    cuda.copyTensor(deviceOutput.view(), hostOutput.view());
    auto cpuOutput = cpu.allocateTensor({1, 2}, DType::FP32);
    cpu.matmulInt8Weights(hostInput.view(), hostWeights.view(), packed.parameters,
                          cpuOutput.view());
    const auto* expected = static_cast<const float*>(cpuOutput.data());
    const auto* actual = static_cast<const float*>(hostOutput.data());
    expect(std::fabs(expected[0] - actual[0]) <= 1.0e-5F &&
               std::fabs(expected[1] - actual[1]) <= 1.0e-5F,
           "CUDA on-the-fly INT8 dequantized GEMM matches the CPU reference");
}

} // namespace

int main() {
    testQuantizerAndCpuReference();
    testArtifactAndResidency();
    testCudaReferenceWhenAvailable();
    if (failures != 0) {
        std::cerr << failures << " Phase 22 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 22 quantized expert tests passed\n";
    return EXIT_SUCCESS;
}
