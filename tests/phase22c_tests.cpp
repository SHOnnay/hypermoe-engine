#include "models/runtime/ExpertDeviceBudget.hpp"
#include "models/runtime/PackedModelRuntime.hpp"
#include "cache/HybridPolicy.hpp"
#include "tests/support/ExpertPipelineFixture.hpp"
#include "backend/CudaBackend.hpp"
#include "profiling/RealModelProfile.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
int failures{};
void expect(bool value, const char* message) {
    if (!value) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template <typename F> void expectThrows(F function, const char* message) {
    try { function(); expect(false, message); }
    catch (const std::exception&) {}
}
constexpr std::size_t MiB = 1024ULL * 1024ULL;
constexpr std::size_t GiB = 1024ULL * MiB;

void planning() {
    using namespace hypermoe::models::runtime;
    const PackedRuntimeConfiguration defaults;
    expect(!defaults.automaticExpertDeviceBudget &&
           defaults.expertDeviceBudgetBytes == 512 * MiB &&
           defaults.expertRamBudgetBytes == 2 * GiB,
           "Phase 22B manual defaults unchanged");
    defaults.validate();
    const hypermoe::backend::MemoryInfo memory{12 * GiB, 7 * GiB, 0};
    ExpertDeviceReservations reservations;
    for (const auto bytes : {512 * MiB, GiB, 2 * GiB, 4 * GiB}) {
        const auto plan = planExpertDeviceBudget(bytes, false, memory, 5 * GiB,
                                                reservations, 6 * MiB);
        expect(plan.effectiveBytes == bytes && plan.reservedBytes == 0,
               "manual 512MiB/1GiB/2GiB/4GiB budgets preserved");
    }
    const auto automatic = planExpertDeviceBudget(512 * MiB, true, memory,
                                                  5 * GiB, reservations, 6 * MiB, 12 * MiB);
    const auto reserved = 256 * MiB + 3 * 512 * MiB + 12 * MiB;
    expect(automatic.automatic && automatic.configuredBytes == 512 * MiB &&
           automatic.reservedBytes == reserved && automatic.effectiveBytes == 7 * GiB - reserved,
           "automatic sizing uses remaining VRAM without double-counting static tensors");
    auto boundary = memory; boundary.freeBytes = reserved + 529U;
    expect(planExpertDeviceBudget(512 * MiB, true, boundary, 5 * GiB,
                                  reservations, 512, 12 * MiB).effectiveBytes == 512,
           "automatic expert budget rounds down to pool alignment");
    boundary.freeBytes = reserved + 128U;
    expectThrows([&] { (void)planExpertDeviceBudget(GiB, true, boundary, 5 * GiB,
                                                    reservations, 1, 12 * MiB); },
                 "sub-alignment remainder cannot admit an expert");
    auto busy = memory; busy.freeBytes = 3 * GiB;
    const auto shared = planExpertDeviceBudget(512 * MiB, true, busy, 5 * GiB,
                                               reservations, 6 * MiB, 12 * MiB);
    expect(shared.effectiveBytes == 3 * GiB - reserved,
           "other GPU users reduce expert capacity");
    auto optimistic = memory; optimistic.freeBytes = 12 * GiB;
    expect(planExpertDeviceBudget(512 * MiB, true, optimistic, 5 * GiB,
                                  reservations, 6 * MiB, 12 * MiB).effectiveBytes == automatic.effectiveBytes,
           "total VRAM minus static allocation is a second upper bound");
    busy.freeBytes = 512 * MiB;
    expectThrows([&] { (void)planExpertDeviceBudget(GiB, false, busy, 5 * GiB,
                                                    reservations, 6 * MiB); },
                 "oversized manual capacity rejected before allocation");
    expectThrows([&] { (void)planExpertDeviceBudget(GiB, true, busy, 5 * GiB,
                                                    reservations, 6 * MiB); },
                 "insufficient reserved headroom rejected without unsigned underflow");
    expectThrows([&] { (void)planExpertDeviceBudget(1, false, memory, 5 * GiB,
                                                    reservations, 6 * MiB); },
                 "largest expert must fit");
    auto invalid = reservations; invalid.stagingPoolBytes = 1;
    expectThrows([&] { (void)planExpertDeviceBudget(GiB, true, memory, 5 * GiB, invalid, 1); },
                 "cannot omit existing pool reserve in automatic mode");
    invalid = reservations; invalid.kvCacheBytes = std::numeric_limits<std::size_t>::max();
    expectThrows([&] { (void)planExpertDeviceBudget(GiB, true, memory, 5 * GiB, invalid, 1); },
                 "reservation addition overflow rejected");
    auto broken = memory; broken.freeBytes = broken.totalBytes + 1;
    expectThrows([&] { (void)planExpertDeviceBudget(GiB, true, broken, 0, reservations, 1); },
                 "invalid device snapshot rejected");
    auto cpu = defaults; cpu.automaticExpertDeviceBudget = true;
    expectThrows([&] { cpu.validate(); }, "automatic VRAM sizing cannot silently affect CPU mode");
}

void scoring() {
    using namespace hypermoe;
    HybridPolicy legacy;
    legacy.onAccess(0);
    legacy.setLayerProbability(0, 1); legacy.setPrefetchConfidence(0, 1);
    expect(std::abs(legacy.score(0) - 1.0) < 1e-12, "legacy hybrid formula unchanged");
    HybridPolicy policy(true, 16);
    for (unsigned i = 0; i < 32; ++i) policy.onAccess(0);
    policy.onAccess(1);
    expect(policy.score(0) > policy.score(1), "hot frequency protects against one-off recency");
    policy.setLayerProbability(2, 1); policy.setPrefetchConfidence(2, 1);
    policy.setLayerProbability(3, 1); policy.setPrefetchConfidence(3, 0);
    expect(policy.score(2) > policy.score(3), "prediction protection requires confidence");
    const auto fresh = policy.score(2);
    for (unsigned i = 0; i < 256; ++i) policy.onAccess(1);
    expect(policy.score(2) < fresh / 1000 && policy.score(1) > policy.score(0),
           "stale prediction and past popularity decay deterministically");
    const std::array<ExpertId, 3> candidates{0, 1, 2};
    const auto victim = policy.selectVictim(MemoryTier::Vram, candidates, {2});
    expect(victim == 0, "active/pinned candidates stay excluded and old hot expert becomes evictable");
    expectThrows([] { HybridPolicy invalid(true, 0); }, "zero aging window rejected");
}

void residency(std::shared_ptr<hypermoe::backend::ComputeBackend> backend = {}) {
    using namespace hypermoe;
    // Real disk reads and real buffers; CpuBackend emulates device memory in
    // always-on tests. The same lifecycle is repeated on CUDA when available.
    for (const auto capacity : {1U, 2U, 4U}) {
        test::ExpertPipelineFixture fixture(false, 12U * capacity, 48, true, {}, backend);
        for (unsigned round = 0; round < 2; ++round) {
            for (ExpertId id = 0; id < 4; ++id) {
                (void)fixture.experts->requestExpert(0, id);
                const auto device = fixture.experts->residentDeviceWeights(0, id);
                expect(device && device->size() == 12, "expert is device-resident before execution");
                if (device) {
                    std::array<std::byte, 12> copy{};
                    fixture.compute->copyFromDevice(copy.data(), device->data(), copy.size());
                    fixture.compute->synchronize();
                    const auto expected = fixture.store->readExpert(0, id, true);
                    expect(std::equal(copy.begin(), copy.end(), expected.begin(), expected.end()),
                           "INT8 layout survives eviction and promotion");
                }
                const auto before = fixture.experts->stats().vramPromotions;
                fixture.experts->adoptDeviceWeights(0, id, device);
                expect(fixture.experts->stats().vramPromotions == before,
                       "resident hit is not counted as another promotion");
                expect(fixture.memory->snapshot().vram.usedBytes <= 12U * capacity,
                       "device budget enforced on all promotions");
            }
        }
        const auto stats = fixture.experts->stats();
        expect(stats.vramPromotions == (capacity == 4 ? 4U : 8U),
               "promotion count distinguishes larger useful residency");
        expect(capacity == 4 ? stats.vramEvictions == 0 : stats.vramEvictions > 0,
               "larger capacity avoids unnecessary evictions");
        expectThrows([&] { fixture.experts->registerExpert(
            {10, 0, 12U * capacity + 1U, QuantizationType::Int8, MemoryTier::Vram}); },
            "expert larger than configured capacity fails cleanly");
    }
}

void policyIntegration() {
    using namespace hypermoe;
    MemoryManager memory(128, 256);
    auto policy = std::make_unique<HybridPolicy>(true, 16);
    auto* scores = policy.get();
    ExpertManager manager(memory, std::move(policy));
    for (ExpertId id = 0; id < 3; ++id) {
        manager.registerExpert({id, 0, 64, QuantizationType::Int8, MemoryTier::Nvme});
    }
    (void)manager.requestExpert(0, 0);
    (void)manager.requestExpert(0, 1);
    manager.updatePrediction(0, 0, 1, 1);
    (void)manager.requestExpert(0, 2);
    expect(manager.findExpert(0, 0)->location == MemoryTier::Vram &&
           manager.findExpert(0, 1)->location == MemoryTier::Ram,
           "existing prediction feedback protects next-use resident under pressure");
    expect(std::abs(manager.residencyScore(0, 0) - scores->score(0)) < 1e-12,
           "residency diagnostics match actual eviction policy score");
    auto fixture = test::ExpertPipelineFixture(true, 12, 48);
    (void)fixture.experts->requestExpert(0, 0);
    auto lease = fixture.experts->acquireResidentExpert(0, 0);
    expectThrows([&] { (void)fixture.experts->requestExpert(0, 1); },
                 "insufficient capacity never evicts an executing expert");
    expect(fixture.experts->findExpert(0, 0)->location == MemoryTier::Vram,
           "active lease remains resident after rejected promotion");
    for (const auto capacity : {1U, 2U, 4U}) {
        test::ExpertPipelineFixture baseline(true, 12U * capacity, 12U * capacity);
        test::ExpertPipelineFixture adaptive(true, 12U * capacity, 12U * capacity,
            true, {}, {}, std::make_unique<HybridPolicy>(true));
        for (unsigned i = 0; i < 24; ++i) {
            baseline.select(0, 1U + i % 3U); adaptive.select(0, 1U + i % 3U);
            const auto a = baseline.execute(); const auto b = adaptive.execute();
            for (std::size_t value = 0; value < 2; ++value) {
                expect(std::abs(static_cast<const float*>(a.output.data())[value] -
                                static_cast<const float*>(b.output.data())[value]) < 1e-5F,
                       "age-aware residency preserves INT8 expert output with Phase 22B overlap");
            }
        }
    }
}

void profile() {
    hypermoe::profiling::RealModelProfile report;
    report.expertDeviceBudgetPlan.configuredBytes = 512 * MiB;
    report.expertDeviceBudgetPlan.effectiveBytes = 4 * GiB;
    report.expertDeviceBudgetPlan.automatic = true;
    report.expertDeviceBudgetBytes = 4 * GiB;
    report.vramPromotions = 9; report.hostToDeviceBytes = 12;
    const auto json = report.toJson();
    expect(json.find("\"vram_promotions\": 9") != std::string::npos &&
           json.find("\"ram_to_vram_transfer_bytes\": 12") != std::string::npos &&
           json.find("\"automatic_expert_device_budget\": true") != std::string::npos &&
           json.find("\"configured_expert_device_budget_bytes\": 536870912") != std::string::npos,
           "report separates configured/effective capacity and movement counters");
}
} // namespace

int main() {
    try {
        planning(); scoring(); residency(); policyIntegration(); profile();
        if (hypermoe::backend::CudaBackend::query().available) {
            residency(std::make_shared<hypermoe::backend::CudaBackend>());
        }
        else std::cout << "SKIP: native CUDA residency (no CUDA device)\n";
    } catch (const std::exception& error) {
        ++failures; std::cerr << "unexpected failure: " << error.what() << '\n';
    }
    std::cout << "Phase 22C failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
