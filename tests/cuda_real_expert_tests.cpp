#include "backend/CudaBackend.hpp"
#include "backend/cuda/CudaRuntimeValidator.hpp"
#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/cache_policy.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "memory/TransferManager.hpp"
#include "models/ExpertWeightMap.hpp"
#include "router/CpuRouterBackend.hpp"
#include "router/Router.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/activation/Activation.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "tensor/precision/DTypeConverter.hpp"
#include "tools/model_convert/ExpertPacker.hpp"
#include "validation/CorrectnessOracle.hpp"

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
                ("hypermoe-phase11-" +
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

std::vector<std::byte> floatBytes(std::span<const float> values) {
    std::vector<std::byte> result(values.size() * sizeof(float));
    if constexpr (std::endian::native == std::endian::little) {
        std::memcpy(result.data(), values.data(), result.size());
    } else {
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto bits = std::bit_cast<std::uint32_t>(values[index]);
            for (unsigned byte = 0; byte < 4; ++byte) {
                result[index * 4 + byte] =
                    static_cast<std::byte>((bits >> (byte * 8U)) & 0xffU);
            }
        }
    }
    return result;
}

void writeSafeTensor(const std::filesystem::path& path,
                     std::string header,
                     std::span<const std::byte> payload) {
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const auto headerSize = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((headerSize >> shift) & 0xffU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    if (!output) throw std::runtime_error("failed writing CUDA test checkpoint");
}

void writeQwenCheckpoint(const std::filesystem::path& root) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Phase11-CUDA-Validation","num_hidden_layers":1,"num_experts":2,"hidden_size":2,"moe_intermediate_size":3,"num_experts_per_tok":1,"norm_topk_prob":true})";

    // Source matrices use Hugging Face output-input layout. The packer transposes
    // them into the input-output layout consumed by both tensor backends.
    const std::vector<float> gateUp{
        // Expert 0 gate, then up.
        1, 0, 0, 1, 1, 1,
        2, 0, 0, 3, 1, -1,
        // Expert 1 gate, then up.
        0.5F, 0, 0, 0.5F, 0.25F, 0.25F,
        1, 0, 0, 1, 0.5F, 0.5F};
    const std::vector<float> down{
        // Expert 0, then expert 1.
        1, 0, 0.5F, 0, 1, -0.5F,
        1, 0, 0, 1, 0, 0};
    const std::vector<float> router{3, 0, 0, 1};
    auto payload = floatBytes(gateUp);
    const auto downBytes = floatBytes(down);
    const auto routerBytes = floatBytes(router);
    payload.insert(payload.end(), downBytes.begin(), downBytes.end());
    payload.insert(payload.end(), routerBytes.begin(), routerBytes.end());
    writeSafeTensor(
        root / "model.safetensors",
        R"({"model.layers.0.mlp.experts.gate_up_proj":{"dtype":"F32","shape":[2,6,2],"data_offsets":[0,96]},"model.layers.0.mlp.experts.down_proj":{"dtype":"F32","shape":[2,2,3],"data_offsets":[96,144]},"model.layers.0.mlp.gate.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[144,160]},"__metadata__":{"format":"pt"}})",
        payload);
}

hypermoe::models::ExpertWeightMap makeMappings(
    const hypermoe::models::ModelManifest& manifest) {
    hypermoe::models::ExpertWeightMap result;
    for (const auto& expert : manifest.experts) {
        const auto add = [&](hypermoe::models::ExpertWeightType type,
                             const hypermoe::models::ProjectionLocation& projection) {
            const auto* tensor = manifest.findTensor(projection.tensorName);
            if (tensor == nullptr) throw std::logic_error("packed tensor is missing");
            result.add(expert.layerId, expert.expertId, type,
                       {tensor->name, projection.shape, tensor->dtype, std::nullopt,
                        projection.offset, projection.size, expert.layerId,
                        expert.expertId});
        };
        add(hypermoe::models::ExpertWeightType::GATE, expert.gate);
        add(hypermoe::models::ExpertWeightType::UP, expert.up);
        add(hypermoe::models::ExpertWeightType::DOWN, expert.down);
    }
    return result;
}

std::vector<float> tensorValues(hypermoe::tensor::TensorView view) {
    if (view.device() != hypermoe::tensor::Device::cpu() ||
        view.dtype() != hypermoe::tensor::DType::FP32) {
        throw std::invalid_argument("test tensor extraction requires CPU FP32");
    }
    const auto* values = static_cast<const float*>(view.data());
    return {values, values + view.shape().elementCount()};
}

std::vector<float> copyToHost(hypermoe::tensor::CudaTensorBackend& cuda,
                              hypermoe::tensor::TensorView device) {
    hypermoe::tensor::CpuTensorBackend cpu;
    auto host = cpu.allocateTensor(device.shape(), device.dtype());
    cuda.copyTensor(device, host);
    return tensorValues(host.view());
}

hypermoe::validation::ExpertOracleTrace captureCudaTrace(
    hypermoe::tensor::CudaTensorBackend& cuda,
    hypermoe::tensor::TensorView input,
    const hypermoe::ExpertMlpWeights& weights) {
    using namespace hypermoe::tensor;
    const Shape intermediate{input.shape().dimensions()[0],
                             weights.gateProjection.shape().dimensions()[1]};
    auto gate = cuda.allocateTensor(intermediate, DType::FP32);
    auto up = cuda.allocateTensor(intermediate, DType::FP32);
    auto activated = cuda.allocateTensor(intermediate, DType::FP32);
    auto gated = cuda.allocateTensor(intermediate, DType::FP32);
    auto output = cuda.allocateTensor(
        {input.shape().dimensions()[0],
         weights.downProjection.shape().dimensions()[1]}, DType::FP32);
    cuda.matmul(input, weights.gateProjection, gate);
    cuda.matmul(input, weights.upProjection, up);
    activation::apply(activation::ActivationType::SiLU, cuda, gate, activated);
    cuda.mul(activated, up, gated);
    cuda.matmul(gated, weights.downProjection, output);
    cuda.synchronize();
    return {copyToHost(cuda, gate), copyToHost(cuda, up),
            copyToHost(cuda, activated), copyToHost(cuda, gated),
            copyToHost(cuda, output)};
}

void testValidationPrecisionAndOracleContracts() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto report = backend::CudaRuntimeValidator::validateAndWrite(
        temporary.path() / "hardware_report.json");
    expect(std::filesystem::exists(temporary.path() / "hardware_report.json") &&
               report.toJson().find("hypermoe.cuda-validation.v1") !=
                   std::string::npos,
           "CUDA validator always emits a structured hardware report");
    if (report.runtimeAvailable) {
        expect(report.passed() && report.device.streamCount == 3,
               "available CUDA runtime validates properties, streams, and events");
    } else {
        expect(report.skipped() && !report.message.empty(),
               "unavailable CUDA runtime reports a clean skip reason");
    }

    const auto native = tensor::precision::DTypeConverter::selectExecutionPlan(
        tensor::DType::FP32, tensor::Device::cuda());
    const auto fp16 = tensor::precision::DTypeConverter::selectExecutionPlan(
        tensor::DType::FP16, tensor::Device::cuda());
    const auto bf16 = tensor::precision::DTypeConverter::selectExecutionPlan(
        tensor::DType::BF16, tensor::Device::cuda());
    expect(native.nativeExecution && !native.conversionRequired &&
               fp16.conversionRequired && bf16.conversionRequired &&
               fp16.executionDType == tensor::DType::FP32,
           "precision policy selects FP32 baseline for FP16/BF16 CUDA storage");
    expectThrows(
        [] {
            (void)tensor::precision::DTypeConverter::selectExecutionPlan(
                tensor::DType::INT8, tensor::Device::cuda());
        },
        "precision policy keeps quantized execution behind an explicit policy");

    validation::ExpertOracleTrace trace;
    trace.gateProjection = {1};
    trace.upProjection = {2};
    trace.activation = {0.7310586F};
    trace.gated = {1.4621172F};
    trace.output = {1.4621172F};
    router::RouterDecision route{0, {1}, {1.0F}};
    const auto comparison = validation::CorrectnessOracle::compareCpuCuda(
        std::vector<float>{2.0F}, std::vector<float>{2.0F}, route, route,
        trace, trace, tensor::DType::FP32);
    expect(comparison.matches() && comparison.finalOutput.maximumAbsoluteError == 0.0F,
           "CPU/CUDA oracle reports stage-level numerical differences");
}

void testRealQwenCudaExpert() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(source);
    writeQwenCheckpoint(source);
    importer::qwen::QwenImporter importer;
    const auto sourceManifest = importer.inspect(source);
    const auto packing = conversion::ExpertPacker{}.pack(
        sourceManifest, source, packed);
    expect(packing.validationPassed && packing.experts == 2,
           "real-format Qwen FP32 artifact imports and packs before CUDA execution");

    auto store = std::make_shared<storage::ExpertStore>(packed);
    const auto manifest = models::ModelManifest::load(packed / "manifest.json");
    const auto mappings = makeMappings(manifest);

    auto cuda = std::make_shared<tensor::CudaTensorBackend>();
    if (!cuda->available()) {
        std::cout << "Phase 11 real CUDA expert test skipped: " << cuda->name()
                  << '\n';
        return;
    }

    tensor::CpuTensorBackend precisionCpu;
    auto hostBf16 = precisionCpu.allocateTensor({2}, tensor::DType::BF16);
    const std::uint16_t bf16Bits[]{0x3f80U, 0xc020U};
    std::memcpy(hostBf16.data(), bf16Bits, sizeof(bf16Bits));
    auto convertedFromHost = tensor::precision::DTypeConverter::toFp32Tensor(
        hostBf16, *cuda);
    auto rawDeviceBf16 = cuda->allocateTensor({2}, tensor::DType::BF16);
    cuda->copyTensor(hostBf16, rawDeviceBf16);
    auto convertedFromDevice = tensor::precision::DTypeConverter::toFp32Tensor(
        rawDeviceBf16, *cuda);
    const auto hostConverted = copyToHost(*cuda, convertedFromHost);
    const auto deviceConverted = copyToHost(*cuda, convertedFromDevice);
    expect(hostConverted == std::vector<float>({1.0F, -2.5F}) &&
               deviceConverted == hostConverted,
           "BF16 staging converts host or CUDA storage to CUDA FP32");

    const auto runtimeValidation = backend::CudaRuntimeValidator::validate();
    expect(runtimeValidation.passed(),
           "CUDA runtime validation passes before execution");
    auto compute = std::make_shared<backend::CudaBackend>();
    auto loader = std::make_shared<storage::DiskLoader>(store);
    auto transfers = std::make_shared<TransferManager>(loader, compute, 2);
    auto profiler = std::make_shared<Profiler>();
    auto scheduler = std::make_shared<scheduler::Scheduler>(transfers, profiler, 2);
    MemoryManager memory(1U << 20U, 1U << 20U);
    ExpertManager experts(memory, std::make_unique<LruCachePolicy>(), transfers);
    for (const auto& record : store->index().records()) {
        experts.registerExpert({record.expert_id, record.layer_id,
                                static_cast<std::size_t>(record.size),
                                QuantizationType::Fp32, MemoryTier::Nvme});
        scheduler->registerExpert(record.layer_id, record.expert_id);
    }

    scheduler::ScheduleRequest request;
    request.layerId = 0;
    request.expertId = 0;
    request.source = MemoryTier::Nvme;
    request.destination = MemoryTier::Vram;
    request.priority = scheduler::TransferPriority::ActiveInference;
    const auto scheduled = scheduler->schedule(request).future().get();
    expect(scheduled.success && scheduled.transfer.cudaTransfer &&
               scheduled.transfer.usedPinnedMemory &&
               scheduled.transfer.ramToVramBytes > 0,
           "scheduler moves a Qwen expert through pinned RAM into CUDA VRAM");
    experts.adoptDeviceWeights(0, 0, scheduled.transfer.deviceBuffer);
    scheduler->acquire(0, 0);

    {
        auto lease = experts.acquireResidentExpert(0, 0);
        expectThrows([&] { experts.moveExpert(0, 0, MemoryTier::Ram); },
                     "active CUDA residency lease prevents expert eviction");
        scheduler::ScheduleRequest blockedEviction;
        blockedEviction.layerId = 0;
        blockedEviction.expertId = 0;
        blockedEviction.source = MemoryTier::Vram;
        blockedEviction.destination = MemoryTier::Ram;
        blockedEviction.priority =
            scheduler::TransferPriority::BackgroundMaintenance;
        blockedEviction.deviceBuffer = scheduled.transfer.deviceBuffer;
        blockedEviction.eviction = true;
        expectThrows([&] { (void)scheduler->schedule(blockedEviction); },
                     "scheduler rejects eviction while CUDA expert is in use");
        const auto expert = experts.findExpert(0, 0);
        auto payload = lease.view({expert->sizeBytes}, tensor::DType::INT8);
        const auto weights = mappings.createViews(
            0, 0, payload, scheduled.transfer.record.offset);

        tensor::CpuTensorBackend cpu;
        auto hostInput = cpu.allocateTensor({1, 2}, tensor::DType::FP32);
        const std::vector<float> inputValues{1.0F, 2.0F};
        std::memcpy(hostInput.data(), inputValues.data(), hostInput.bytes());
        auto deviceInput = cuda->allocateTensor({1, 2}, tensor::DType::FP32);
        auto deviceOutput = cuda->allocateTensor({1, 2}, tensor::DType::FP32);
        cuda->copyTensor(hostInput, deviceInput);
        ExpertMlpExecutor executor(cuda);
        executor.execute(deviceInput, weights, deviceOutput);
        const auto actualOutput = copyToHost(*cuda, deviceOutput);

        const std::vector<float> gate{1, 0, 1, 0, 1, 1};
        const std::vector<float> up{2, 0, 1, 0, 3, -1};
        const std::vector<float> down{1, 0, 0, 1, 0.5F, -0.5F};
        const auto cpuTrace = validation::CorrectnessOracle::expertMlpTrace(
            inputValues, 1, 2, gate, up, down, 3);
        const auto cudaTrace = captureCudaTrace(*cuda, deviceInput, weights);

        const std::vector<float> routerWeights{3, 0, 0, 1};
        const auto cpuLogits = validation::CorrectnessOracle::routerLogits(
            inputValues, routerWeights, 2);
        auto hostRouter = cpu.allocateTensor({2, 2}, tensor::DType::FP32);
        std::memcpy(hostRouter.data(), routerWeights.data(), hostRouter.bytes());
        auto deviceRouter = cuda->allocateTensor({2, 2}, tensor::DType::FP32);
        auto deviceLogits = cuda->allocateTensor({1, 2}, tensor::DType::FP32);
        cuda->copyTensor(hostRouter, deviceRouter);
        cuda->matmul(deviceInput, deviceRouter, deviceLogits);
        const auto cudaLogits = copyToHost(*cuda, deviceLogits);
        const auto cpuRouting = validation::CorrectnessOracle::selectFromLogits(
            0, cpuLogits, manifest.router.config);
        const auto cudaRouting = validation::CorrectnessOracle::selectFromLogits(
            0, cudaLogits, manifest.router.config);
        const auto report = validation::CorrectnessOracle::compareCpuCuda(
            cpuLogits, cudaLogits, cpuRouting, cudaRouting, cpuTrace, cudaTrace,
            tensor::DType::FP32);
        const auto final = validation::CorrectnessOracle::compare(
            actualOutput, cpuTrace.output,
            validation::CorrectnessOracle::toleranceFor(tensor::DType::FP32));
        expect(report.matches() && final.matches &&
                   report.finalOutput.maximumAbsoluteError <= 1.0e-5F,
               "Qwen router and every CUDA expert stage match the CPU oracle");
    }

    scheduler->release(0, 0);
    scheduler::ScheduleRequest eviction;
    eviction.layerId = 0;
    eviction.expertId = 0;
    eviction.source = MemoryTier::Vram;
    eviction.destination = MemoryTier::Ram;
    eviction.priority = scheduler::TransferPriority::BackgroundMaintenance;
    eviction.deviceBuffer = scheduled.transfer.deviceBuffer;
    eviction.eviction = true;
    const auto evicted = scheduler->schedule(std::move(eviction)).future().get();
    expect(evicted.success && evicted.transfer.buffer,
           "scheduler downloads an evictable expert after execution completes");
    experts.adoptHostWeights(0, 0, evicted.transfer.buffer);
    expect(experts.findExpert(0, 0)->location == MemoryTier::Ram &&
               !experts.residentDeviceWeights(0, 0) &&
               experts.residentWeights(0, 0) == evicted.transfer.buffer &&
               scheduler->state(0, 0).currentLocation == MemoryTier::Ram,
           "scheduler and expert manager adopt one completed eviction buffer");
    expect(profiler->snapshot().queueWaitSamples > 0,
           "CUDA transfer path records scheduler queue wait behavior");
}

} // namespace

int main() {
    testValidationPrecisionAndOracleContracts();
    testRealQwenCudaExpert();
    if (failures != 0) {
        std::cerr << failures << " Phase 11 assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 11 CUDA real expert tests passed\n";
    return EXIT_SUCCESS;
}
