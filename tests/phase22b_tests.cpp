#include "tests/support/ExpertPipelineFixture.hpp"
#include "backend/CudaBackend.hpp"
#include "models/runtime/PackedModelRuntime.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
int failures{};
void expect(bool value, const char* message) {
    if (!value) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template <typename Function> void expectThrows(Function function) {
    try { function(); expect(false, "invalid budget must fail"); }
    catch (const std::exception&) {}
}

class ObservedCpuBackend final : public hypermoe::tensor::TensorBackend {
    hypermoe::tensor::CpuTensorBackend cpu;
public:
    std::function<void()> projection;
    std::string_view name() const noexcept override { return "observed_cpu"; }
    hypermoe::tensor::Device device() const noexcept override { return cpu.device(); }
    bool available() const noexcept override { return true; }
    hypermoe::tensor::Tensor allocateTensor(const hypermoe::tensor::Shape& s,
                                            hypermoe::tensor::DType d) override { return cpu.allocateTensor(s, d); }
    void copyTensor(hypermoe::tensor::TensorView a, hypermoe::tensor::TensorView b) override { cpu.copyTensor(a, b); }
    void matmul(hypermoe::tensor::TensorView a, hypermoe::tensor::TensorView b,
                hypermoe::tensor::TensorView c) override { if (projection) projection(); cpu.matmul(a, b, c); }
    void matmulInt8Weights(hypermoe::tensor::TensorView a, hypermoe::tensor::TensorView b,
        const hypermoe::tensor::quantization::QuantizationParameters& q,
        hypermoe::tensor::TensorView c) override { if (projection) projection(); cpu.matmulInt8Weights(a, b, q, c); }
    void add(hypermoe::tensor::TensorView a, hypermoe::tensor::TensorView b,
              hypermoe::tensor::TensorView c) override { cpu.add(a, b, c); }
    void mul(hypermoe::tensor::TensorView a, hypermoe::tensor::TensorView b,
              hypermoe::tensor::TensorView c) override { cpu.mul(a, b, c); }
    hypermoe::tensor::Tensor reshape(const hypermoe::tensor::Tensor& t,
                                     hypermoe::tensor::Shape s) override { return cpu.reshape(t, std::move(s)); }
    void synchronize() override { cpu.synchronize(); }
};

void budgets() {
    using Configuration = hypermoe::models::runtime::PackedRuntimeConfiguration;
    const Configuration defaults;
    expect(defaults.expertDeviceBudgetBytes == 512U * 1024U * 1024U &&
           defaults.expertRamBudgetBytes == 2ULL * 1024ULL * 1024ULL * 1024ULL,
           "existing budgets unchanged");
    for (const auto gib : {1U, 2U, 4U}) {
        expect(Configuration::parseBudgetBytes(std::to_string(gib) + "GiB") ==
                   static_cast<std::size_t>(gib) * 1024ULL * 1024ULL * 1024ULL,
               "GiB budget configuration");
    }
    expect(Configuration::parseBudgetBytes("512MB") == 512000000U &&
           Configuration::parseBudgetBytes("512MiB") == defaults.expertDeviceBudgetBytes,
           "decimal and binary units distinct");
    for (const auto invalid : {"", "0", "-1GiB", "1.5GiB", "4GBx", "18446744073709551615GiB"}) {
        expectThrows([&] { (void)Configuration::parseBudgetBytes(invalid); });
    }
    auto wide = hypermoe::models::runtime::ModelArchitecture{};
    wide.layerCount = 1; wide.hiddenDimension = 2048;
    wide.attentionHeads = 32; wide.keyValueHeads = 4;
    wide.headDimension = 128; wide.projectionHeadDimension = 128;
    wide.expertCount = 128; wide.topK = 8;
    wide.validate();
    wide.headDimension = 64; wide.validate();
    expect(true, "wide and distinct projection-head dimensions accepted with GQA");
}

void cpuPipeline() {
    using namespace hypermoe;
    for (const bool int8 : {false, true}) {
        const std::size_t size = int8 ? 12U : 48U;
        for (const auto capacity : {1U, 2U, 4U}) {
            test::ExpertPipelineFixture serial(false, size * capacity, size * capacity, int8);
            test::ExpertPipelineFixture overlap(true, size * capacity, size * capacity, int8);
            for (unsigned iteration = 0; iteration < 12; ++iteration) {
                const ExpertId first = iteration % 4;
                const ExpertId second = (iteration + 1) % 4;
                serial.select(first, second); overlap.select(first, second);
                const auto a = serial.execute(); const auto b = overlap.execute();
                const auto* av = static_cast<const float*>(a.output.data());
                const auto* bv = static_cast<const float*>(b.output.data());
                expect(std::abs(av[0] - bv[0]) < 1e-5F && std::abs(av[1] - bv[1]) < 1e-5F,
                       "serial and overlap expert outputs match");
                expect(overlap.memory->snapshot().ram.usedBytes <= size * capacity &&
                       overlap.memory->snapshot().vram.usedBytes == 0,
                       "CPU path bounded RAM and no VRAM residency");
                expect(!overlap.scheduler->cachedTransfer(0, second).has_value(),
                       "adoption transfers scheduler ownership to manager");
            }
            expect(capacity == 4 || overlap.experts->stats().ramEvictions != 0,
                   "limited RAM budget causes eviction and reload");
        }
    }
}

void ordering(bool warmPrediction) {
    using namespace hypermoe;
    auto backend = std::make_shared<ObservedCpuBackend>();
    test::ExpertPipelineFixture fixture(true, 48, 48, true, backend);
    std::mutex mutex;
    std::condition_variable condition;
    bool firstCompute{}, nextStarted{}, nextReady{}, timedOut{};
    bool firstProjection = true;
    const auto subscription = fixture.scheduler->events().subscribe([&](const scheduler::RuntimeEvent& event) {
        if (event.expertId != 1) return;
        std::unique_lock lock(mutex);
        if (event.type == scheduler::RuntimeEventType::TransferStarted) {
            nextStarted = true;
            condition.notify_all();
            if (!condition.wait_for(lock, std::chrono::seconds(3), [&] { return firstCompute; })) timedOut = true;
        }
        if (event.type == scheduler::RuntimeEventType::ExpertReady) {
            nextReady = true;
            condition.notify_all();
        }
    });
    backend->projection = [&] {
        if (!firstProjection) return;
        firstProjection = false;
        std::unique_lock lock(mutex);
        firstCompute = true;
        condition.notify_all();
        if (!condition.wait_for(lock, std::chrono::seconds(3), [&] { return nextStarted && nextReady; })) timedOut = true;
    };
    scheduler::ScheduleHandle warm;
    if (warmPrediction) warm = fixture.scheduler->prefetch({0, 1, 1.0, 1.0});
    const auto result = fixture.execute();
    expect(!timedOut && firstCompute && nextReady,
           "B transfer completes while A compute is active, without all-expert barrier");
    expect(fixture.scheduler->state(0, 1).state == scheduler::ExpertLifecycleState::Ready &&
           result.expertOutputs.size() == 2,
           "expert is ready before execution and released afterward");
    expect(fixture.scheduler->events().unsubscribe(subscription), "event subscription cleanup");
}

void deviceBudgets() {
    using namespace hypermoe;
    // Physical buffers backed by CPU for always-on eviction/promotion coverage.
    for (const auto capacity : {1U, 2U, 4U}) {
        test::ExpertPipelineFixture fixture(true, 12U * capacity, 48);
        for (ExpertId id = 0; id < 4; ++id) {
            (void)fixture.experts->requestExpert(0, id);
            expect(fixture.memory->snapshot().vram.usedBytes <= 12U * capacity,
                   "device budget is enforced on promotion");
        }
        expect(fixture.memory->snapshot().vram.usedBytes == 12U * capacity,
               "different limits change device residency capacity");
        expect(capacity == 4 || fixture.experts->stats().vramEvictions != 0,
               "small device budget evicts experts");
        (void)fixture.experts->requestExpert(0, 0);
        expect(fixture.experts->findExpert(0, 0)->location == MemoryTier::Vram,
               "demoted expert can be promoted");
    }
    test::ExpertPipelineFixture protectedExpert(true, 24, 48);
    (void)protectedExpert.experts->requestExpert(0, 0);
    (void)protectedExpert.experts->requestExpert(0, 1);
    const auto lease = protectedExpert.experts->acquireResidentExpert(0, 0);
    protectedExpert.experts->prepareResidency(0, 2, MemoryTier::Vram);
    expect(protectedExpert.experts->findExpert(0, 0)->location == MemoryTier::Vram &&
           protectedExpert.experts->findExpert(0, 1)->location == MemoryTier::Ram,
           "lookahead room preparation cannot evict an in-use expert");
}

void warmPrefetch() {
    using namespace hypermoe;
    test::ExpertPipelineFixture fixture(true, 48, 48);
    for (ExpertId id = 0; id < 2; ++id) {
        auto handle = fixture.scheduler->prefetch({0, id, 1.0, 1.0});
        expect(handle.future().get().success, "warm prediction completes");
    }
    const auto loaded = fixture.profiler->snapshot().nvmeBytes;
    (void)fixture.execute();
    expect(fixture.profiler->snapshot().nvmeBytes == loaded && loaded == 24,
           "CPU consumes prefetched host weights without duplicate disk reads");
    expect(fixture.profiler->snapshot().prefetchHits == 2,
           "warm prediction consumption records usefulness");
    expect(!fixture.scheduler->cachedTransfer(0, 0) &&
           !fixture.scheduler->cachedTransfer(0, 1),
           "warm result ownership released after adoption");
    test::ExpertPipelineFixture bounded(true, 48, 12);
    for (ExpertId id = 0; id < 4; ++id) {
        auto handle = bounded.scheduler->prefetch({0, id, 1.0, 1.0});
        expect(handle.future().get().success, "bounded speculative request completes");
    }
    std::size_t cachedBytes{};
    for (ExpertId id = 0; id < 4; ++id) {
        if (const auto cached = bounded.scheduler->cachedTransfer(0, id)) cachedBytes += cached->record.size;
    }
    expect(cachedBytes <= 12, "speculative scheduler ownership has a configured bound");
}

void cudaPipeline() {
    using namespace hypermoe;
    auto cuda = std::make_shared<tensor::CudaTensorBackend>();
    if (!cuda->available() || !cuda->nativeKernelsAvailable()) {
        std::cout << "SKIP: native CUDA overlap correctness (CUDA unavailable)\n";
        return;
    }
    auto transfer = std::make_shared<backend::CudaBackend>();
    test::ExpertPipelineFixture cpu(false, 48, 48);
    test::ExpertPipelineFixture gpu(true, 24, 48, true, cuda, transfer);
    for (unsigned index = 0; index < 8; ++index) {
        cpu.select(index % 4, (index + 1) % 4);
        gpu.select(index % 4, (index + 1) % 4);
        const auto a = cpu.execute(); const auto b = gpu.execute();
        tensor::CpuTensorBackend host;
        auto values = host.allocateTensor(b.output.shape(), b.output.dtype());
        cuda->copyTensor(b.output.view(), values.view());
        for (std::size_t value = 0; value < 2; ++value) {
            expect(std::abs(static_cast<const float*>(a.output.data())[value] -
                            static_cast<const float*>(values.data())[value]) < 1e-5F,
                   "CUDA INT8 lookahead matches CPU output under eviction");
        }
        expect(gpu.memory->snapshot().vram.usedBytes <= 24, "CUDA logical expert budget");
    }
}
} // namespace

int main() {
    try { budgets(); cpuPipeline(); ordering(false); ordering(true); deviceBudgets(); warmPrefetch(); cudaPipeline(); }
    catch (const std::exception& error) { ++failures; std::cerr << error.what() << '\n'; }
    if (failures) return EXIT_FAILURE;
    std::cout << "All Phase 22B tests passed\n";
    return EXIT_SUCCESS;
}
