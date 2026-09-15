#include "backend/cuda/Int8GemmPlan.hpp"
#include "experts/ExpertExecutor.hpp"
#include "models/runtime/PackedModelRuntime.hpp"
#include "profiling/GpuEventQueue.hpp"
#include "profiling/Profiler.hpp"
#include "profiling/RealModelProfile.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/attention/CudaAttention.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {
int failures{};
void expect(bool value, std::string_view message) {
    if (!value) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template<class F> void expectThrows(F&& action, std::string_view message) {
    try { action(); expect(false, message); } catch (const std::exception&) {}
}

void plans() {
    using namespace hypermoe::backend::cuda;
    const auto gate = planInt8Gemm(1, 2048, 768);
    expect(gate.implementation == Int8GemmMode::Cooperative && gate.partitions == 4 &&
           gate.blocks == 96 && gate.scratchElements == 3072, "Qwen gate/up launch mapping");
    const auto down = planInt8Gemm(1, 768, 2048);
    expect(down.partitions == 2 && down.blocks == 128, "Qwen down launch mapping");
    expect(planInt8Gemm(1, 2048, 768, Int8GemmMode::Reference).blocks == 3,
           "reference mode preserved");
    expect(planInt8Gemm(1, 8, 8).implementation == Int8GemmMode::Reference,
           "tiny projections keep reference dispatch");
    expect(planInt8Gemm(2, 257, 65).scratchElements == 0, "single partition needs no scratch");
    expect(planInt8Gemm(1, 8193, 65).partitions == 8, "bounded scratch partitions");
    expectThrows([] { (void)planInt8Gemm(0, 10, 10); }, "zero dimensions rejected");
    expectThrows([] { (void)planInt8Gemm(2, std::numeric_limits<std::size_t>::max(), 2); }, "overflow rejected");
    expectThrows([] { (void)planInt8Gemm(1, 1, 1, static_cast<Int8GemmMode>(99)); }, "bad mode rejected");
    hypermoe::models::runtime::PackedRuntimeConfiguration configuration;
    configuration.int8GemmMode = Int8GemmMode::Reference;
    configuration.validate();
    expect(configuration.expertDeviceBudgetBytes == 512U * 1024U * 1024U,
           "existing residency default unchanged");
    configuration.int8GemmMode = static_cast<Int8GemmMode>(99);
    expectThrows([&] { configuration.validate(); }, "runtime validates mode on CPU too");
    hypermoe::models::runtime::ModelArchitecture architecture;
    architecture.layerCount = 1; architecture.hiddenDimension = 2048;
    architecture.attentionHeads = 32; architecture.keyValueHeads = 4;
    architecture.headDimension = 128; architecture.projectionHeadDimension = 128;
    architecture.expertCount = 128; architecture.topK = 8;
    architecture.validate(); // Wide attention width 4096, not hidden width 2048.
    architecture.headDimension = 64;
    architecture.validate();
    expect(architecture.projectionHeadDimension == 128, "projection metadata remains separate");
}

struct FakeEvents {
    struct Event { std::atomic_bool ready{}; };
    std::vector<std::unique_ptr<Event>> events;
    std::size_t waits{};
    std::size_t destroyed{};
    bool failQuery{};
    bool failRecord{};
    hypermoe::profiling::GpuEventApi api() {
        return {
            [this](bool) -> void* { events.push_back(std::make_unique<Event>()); return events.back().get(); },
            [this](void*, void*) { if (failRecord) throw std::runtime_error("record error"); },
            [this](void* event) { if (failQuery) throw std::runtime_error("query error"); return static_cast<Event*>(event)->ready.load(); },
            [](void*, void*) { return 0.25F; },
            [this](void* event) { if (event) ++destroyed; },
            [this](void*) { ++waits; complete(); }
        };
    }
    void complete() { for (auto& event : events) event->ready.store(true); }
};

void events() {
    using namespace hypermoe;
    FakeEvents fake;
    auto profiler = std::make_shared<Profiler>();
    profiling::GpuEventQueue queue(fake.api(), profiler);
    auto owner = std::make_shared<int>(12);
    std::weak_ptr<int> weak = owner;
    {
        auto scope = queue.begin(profiling::GpuOperation::ExpertInt8Gemm, nullptr, {owner});
        scope.finish();
    }
    owner.reset(); queue.collect();
    expect(!weak.expired() && queue.pending() == 1 && fake.waits == 0,
           "in-flight owners retained; timing collection never waits");
    fake.failQuery = true;
    expectThrows([&] { queue.collect(); }, "query errors not hidden");
    expect(!weak.expired(), "query failure retains in-flight storage");
    fake.failQuery = false;
    // Simulate asynchronous completion from a device worker, with atomic readiness.
    std::thread worker([&] { fake.complete(); }); worker.join(); queue.collect();
    expect(weak.expired() && queue.pending() == 0 && fake.waits == 0 && fake.destroyed == 2,
           "release only after completion, with no profiling synchronization");
    auto metrics = profiler->snapshot();
    expect(metrics.cudaExpertGemmTime == std::chrono::microseconds(250) &&
           metrics.cudaKernelTime == metrics.cudaExpertGemmTime, "typed GPU leaf accounting");
    expect(metrics.matmulTime == metrics.cudaExpertGemmTime && metrics.kernelTime == metrics.cudaKernelTime,
           "legacy aggregate GEMM/kernel metrics remain compatible");
    profiler->recordGpuTime(profiling::GpuOperation::ExpertRegion, std::chrono::milliseconds(1));
    expect(profiler->snapshot().cudaKernelTime == metrics.cudaKernelTime,
           "inclusive regions do not double count kernel spans");
    expect(profiler->toJson().find("\"gpu_utilization_percent\": null") != std::string::npos,
           "unmeasured GPU utilization is null");
    profiling::RealModelProfile profile;
    profile.gpuTimings = profiler->snapshot();
    profile.totalWallTime = std::chrono::milliseconds(2);
    const auto json = profile.toJson();
    expect(json.find("\"cuda_expert_gemm_time_ms\": 0.250000") != std::string::npos &&
           json.find("\"cuda_expert_execution_time_ms\": 1.000000") != std::string::npos &&
           json.find("\"cuda_leaf_span_wall_ratio\": 0.125000") != std::string::npos,
           "real-model report distinguishes inclusive and leaf GPU spans");
    fake.failRecord = true;
    expectThrows([&] { auto scope = queue.begin(profiling::GpuOperation::Activation, nullptr); scope.finish(); },
                 "record failure surfaces");
    expect(fake.waits == 1, "exception-only stream lifetime barrier");
    fake.failRecord = false;
    FakeEvents unprofiled;
    std::weak_ptr<int> retained;
    {
        profiling::GpuEventQueue plain(unprofiled.api());
        auto storage = std::make_shared<int>(7); retained = storage;
        auto scope = plain.begin(profiling::GpuOperation::Int8Gemm, nullptr, {storage});
        scope.finish(); storage.reset(); plain.collect();
        expect(!retained.expired(), "ownership protection works without profiling");
    }
    expect(retained.expired() && unprofiled.waits == 1, "shutdown drains before releasing storage");
}

// Host emulation checks partition/tile coverage and reduction order; it is not
// a substitute for native CUDA execution or device race checking.
std::vector<float> emulateCooperative(const float* input, const std::int8_t* weights,
    std::size_t rows, std::size_t inner, std::size_t columns, float scale, std::int32_t zeroPoint) {
    const auto plan = hypermoe::backend::cuda::planInt8Gemm(rows, inner, columns,
        hypermoe::backend::cuda::Int8GemmMode::Cooperative);
    std::vector<float> result(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t part = 0; part < plan.partitions; ++part) {
                const auto base = inner / plan.partitions;
                const auto remainder = inner % plan.partitions;
                const auto begin = part * base + std::min(part, remainder);
                const auto end = begin + base + static_cast<std::size_t>(part < remainder);
                std::array<float, 8> sums{};
                for (auto tile = begin; tile < end; tile += 256) {
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        for (auto k = lane; k < 256 && k < end - tile; k += 8) {
                            const auto weight = static_cast<float>(static_cast<int>(weights[(tile + k) * columns + column]) - zeroPoint) * scale;
                            sums[lane] += input[row * inner + tile + k] * weight;
                        }
                    }
                }
                for (std::size_t stride = 4; stride != 0; stride /= 2) {
                    for (std::size_t lane = 0; lane < stride; ++lane) sums[lane] += sums[lane + stride];
                }
                result[row * columns + column] += sums[0];
            }
        }
    }
    return result;
}

void gemmCorrectness() {
    using namespace hypermoe;
    using namespace tensor;
    using backend::cuda::Int8GemmMode;
    auto profiler = std::make_shared<Profiler>();
    CudaTensorBackend optimized(0, profiler, Int8GemmMode::Cooperative);
    if (!optimized.nativeKernelsAvailable()) {
        std::cout << "SKIP: native CUDA GEMM comparisons (no native CUDA backend)\n";
    }
    CudaTensorBackend reference(0, {}, Int8GemmMode::Reference);
    CpuTensorBackend cpu;
    const std::array<std::array<std::size_t, 3>, 8> sizes{{
        {1, 8, 8}, {1, 257, 65}, {2, 513, 33}, {1, 1031, 31},
        {1, 2048, 768}, {1, 768, 2048}, {3, 129, 7}, {1, 4097, 67}}};
    for (const auto& dims : sizes) {
        const auto [rows, inner, columns] = dims;
        auto a = cpu.allocateTensor({rows, inner}, DType::FP32);
        auto b = cpu.allocateTensor({inner, columns}, DType::INT8);
        auto c = cpu.allocateTensor({rows, columns}, DType::FP32);
        auto* input = static_cast<float*>(a.data());
        auto* weights = static_cast<std::int8_t*>(b.data());
        for (std::size_t i = 0; i < rows * inner; ++i) input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 29.0F;
        for (std::size_t i = 0; i < inner * columns; ++i) weights[i] = static_cast<std::int8_t>(static_cast<int>((i * 13) % 255) - 127);
        const quantization::QuantizationParameters parameters{0.003F, -3};
        cpu.matmulInt8Weights(a, b, parameters, c);
        const auto emulated = emulateCooperative(input, weights, rows, inner, columns, parameters.scale, parameters.zeroPoint);
        const auto* expected = static_cast<const float*>(c.data());
        for (std::size_t i = 0; i < rows * columns; ++i) {
            expect(std::abs(emulated[i] - expected[i]) <= 1.0e-5F + 1.0e-5F * std::abs(expected[i]),
                   "host emulation of cooperative mapping vs CPU reference");
        }
        if (!optimized.nativeKernelsAvailable()) continue;
        auto run = [&](CudaTensorBackend& gpu) {
            auto ga = gpu.allocateTensor(a.shape(), a.dtype());
            auto gb = gpu.allocateTensor(b.shape(), b.dtype());
            auto gc = gpu.allocateTensor(c.shape(), c.dtype());
            gpu.copyTensor(a, ga); gpu.copyTensor(b, gb);
            const auto before = gpu.backendStats().synchronizationCount;
            gpu.matmulInt8Expert(ga, gb, parameters, gc);
            expect(gpu.backendStats().synchronizationCount == before, "GEMM does not synchronize for profiling");
            // Destroy input owners before completion: event queue must retain both.
            ga = {}; gb = {};
            auto result = cpu.allocateTensor(c.shape(), c.dtype());
            gpu.copyTensor(gc, result);
            return result;
        };
        auto actual = run(optimized); auto old = run(reference);
        const auto* current = static_cast<const float*>(actual.data());
        const auto* previous = static_cast<const float*>(old.data());
        for (std::size_t i = 0; i < rows * columns; ++i) {
            const auto tolerance = 1.0e-5F + 1.0e-5F * std::abs(expected[i]);
            expect(std::isfinite(current[i]) && std::abs(current[i] - expected[i]) <= tolerance,
                   "cooperative INT8 vs CPU tolerance");
            expect(std::abs(current[i] - previous[i]) <= tolerance, "cooperative vs scalar CUDA tolerance");
            // Independent double-accumulation oracle (same quantized representation).
            const auto row = i / columns; const auto col = i % columns;
            double sum{};
            for (std::size_t k = 0; k < inner; ++k) {
                const auto weight = static_cast<float>(static_cast<int>(weights[k * columns + col]) - parameters.zeroPoint) * parameters.scale;
                sum += static_cast<double>(input[row * inner + k]) * weight;
            }
            expect(std::abs(static_cast<double>(current[i]) - sum) <= static_cast<double>(tolerance),
                   "cooperative INT8 vs independent affine oracle");
        }
    }
    if (optimized.nativeKernelsAvailable()) {
        optimized.synchronizeExecution();
        expect(profiler->snapshot().cudaExpertGemmTime.count() > 0, "native expert GEMM event time recorded");
    }
}

void wideAttention() {
    using namespace hypermoe;
    using namespace tensor;
    CpuTensorBackend cpu;
    std::vector<Tensor> host;
    const auto make = [&](Shape shape, bool norm = false) {
        auto value = cpu.allocateTensor(shape, DType::FP32);
        auto* data = static_cast<float*>(value.data());
        for (std::size_t i = 0; i < shape.elementCount(); ++i) {
            data[i] = norm ? 1.0F : 0.04F * static_cast<float>(static_cast<int>((i * 7) % 13) - 6);
        }
        host.push_back(std::move(value));
    };
    make({2, 4}); make({4, 8}); make({4, 4}); make({4, 4}); make({8, 4}); make({4}, true); make({4}, true);
    transformer::attention::AttentionConfiguration config;
    config.headCount = 2; config.keyValueHeadCount = 1;
    config.headDimension = 4; config.projectionHeadDimension = 4;
    config.causal = true; config.rotaryEmbedding = true;
    auto cpuBackend = std::make_shared<CpuTensorBackend>();
    auto expected = transformer::attention::CpuAttention(cpuBackend).execute(host[0],
        {host[1], host[2], host[3], host[4], host[5], host[6]}, config);
    expect(expected.query.shape() == Shape{2, 8} && expected.output.shape() == Shape{2, 4},
           "CPU wide attention/GQA with QK norm and RoPE preserved");
    auto profiler = std::make_shared<Profiler>();
    auto gpu = std::make_shared<CudaTensorBackend>(0, profiler);
    if (!gpu->nativeKernelsAvailable()) {
        std::cout << "SKIP: native CUDA wide-attention comparison\n";
        return;
    }
    std::vector<Tensor> device;
    for (const auto& value : host) {
        auto copied = gpu->allocateTensor(value.shape(), value.dtype());
        gpu->copyTensor(value, copied); device.push_back(std::move(copied));
    }
    const auto before = gpu->backendStats().synchronizationCount;
    auto actual = transformer::attention::CudaAttention(gpu).execute(device[0],
        {device[1], device[2], device[3], device[4], device[5], device[6]}, config);
    expect(gpu->backendStats().synchronizationCount == before,
           "native FP32 attention operations do not wait for profiling");
    const auto compare = [&](const Tensor& reference, const Tensor& current) {
        auto result = cpu.allocateTensor(current.shape(), current.dtype());
        gpu->copyTensor(current, result);
        const auto* a = static_cast<const float*>(reference.data());
        const auto* b = static_cast<const float*>(result.data());
        for (std::size_t i = 0; i < reference.shape().elementCount(); ++i) {
            expect(std::isfinite(b[i]) && std::abs(a[i] - b[i]) <= 1.0e-5F,
                   "wide CUDA attention intermediate matches CPU");
        }
    };
    compare(expected.query, actual.query); compare(expected.key, actual.key);
    compare(expected.probabilities, actual.probabilities); compare(expected.output, actual.output);
    const auto timing = profiler->snapshot();
    expect(timing.cudaAttentionTime.count() > 0 && timing.cudaAttentionCoreTime.count() > 0,
           "native attention inclusive/core event timing recorded");
}

void int8Mlp() {
    using namespace hypermoe;
    using namespace tensor;
    auto cpu = std::make_shared<CpuTensorBackend>();
    auto input = cpu->allocateTensor({1, 513}, DType::FP32);
    auto gate = cpu->allocateTensor({513, 67}, DType::INT8);
    auto up = cpu->allocateTensor({513, 67}, DType::INT8);
    auto down = cpu->allocateTensor({67, 513}, DType::INT8);
    auto expected = cpu->allocateTensor({1, 513}, DType::FP32);
    auto* x = static_cast<float*>(input.data());
    for (std::size_t i = 0; i < 513; ++i) x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 17.0F;
    for (auto* value : {&gate, &up, &down}) {
        auto* q = static_cast<std::int8_t*>(value->data());
        for (std::size_t i = 0; i < value->shape().elementCount(); ++i) q[i] = static_cast<std::int8_t>(static_cast<int>((i * 13) % 255) - 127);
    }
    const quantization::QuantizationParameters parameters{0.001F, 0};
    ExpertMlpExecutor(cpu).execute(input, {gate, up, down, parameters, parameters, parameters}, expected);
    auto profiler = std::make_shared<Profiler>();
    auto gpu = std::make_shared<CudaTensorBackend>(0, profiler, backend::cuda::Int8GemmMode::Cooperative);
    if (!gpu->nativeKernelsAvailable()) {
        std::cout << "SKIP: native CUDA INT8 MLP/lifetime comparison\n";
        return;
    }
    const auto copy = [&](const Tensor& host) {
        auto result = gpu->allocateTensor(host.shape(), host.dtype()); gpu->copyTensor(host, result); return result;
    };
    auto gi = copy(input); auto gg = copy(gate); auto gu = copy(up); auto gd = copy(down);
    auto result = gpu->allocateTensor(expected.shape(), expected.dtype());
    const auto before = gpu->backendStats().synchronizationCount;
    ExpertMlpExecutor(gpu, activation::ActivationType::SiLU, profiler).execute(gi,
        {gg, gu, gd, parameters, parameters, parameters}, result);
    expect(gpu->backendStats().synchronizationCount == before, "INT8 MLP enqueues without profiler waits");
    gi = {}; gg = {}; gu = {}; gd = {}; // End events must retain external and intermediate owners.
    auto actual = cpu->allocateTensor(expected.shape(), expected.dtype()); gpu->copyTensor(result, actual);
    const auto* a = static_cast<const float*>(expected.data());
    const auto* b = static_cast<const float*>(actual.data());
    for (std::size_t i = 0; i < 513; ++i) {
        expect(std::isfinite(b[i]) && std::abs(a[i] - b[i]) <= 1.0e-5F + 1.0e-5F * std::abs(a[i]),
               "cooperative INT8 MLP matches CPU with released input/weight owners");
    }
    const auto timing = profiler->snapshot();
    expect(timing.cudaExpertGemmTime.count() > 0 && timing.cudaActivationTime.count() > 0 &&
           timing.cudaExpertExecutionTime.count() > 0, "INT8 expert leaf/region GPU timing recorded");
}
} // namespace

int main() {
    try { plans(); events(); gemmCorrectness(); wideAttention(); int8Mlp(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; ++failures; }
    if (failures) return 1;
    std::cout << "Phase 23 tests passed\n";
    return 0;
}
