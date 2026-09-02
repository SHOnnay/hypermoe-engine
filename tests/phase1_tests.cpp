#include "hypermoe/experts/expert_manager.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace {

constexpr std::size_t MiB = 1024 * 1024;
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void testMemoryLimitsAndRelease() {
    hypermoe::MemoryManager memory(8 * MiB, 16 * MiB);
    const auto allocation = memory.allocate(hypermoe::MemoryTier::Vram, 6 * MiB, "test");
    expect(allocation.has_value(), "valid VRAM allocation succeeds");
    expect(!memory.allocate(hypermoe::MemoryTier::Vram, 3 * MiB),
           "VRAM limit is enforced");
    expect(memory.snapshot().vram.usedBytes == 6 * MiB, "VRAM usage is tracked");
    expect(memory.release(allocation->id), "known allocation releases once");
    expect(!memory.release(allocation->id), "double release is rejected");
    expect(memory.snapshot().vram.usedBytes == 0, "release restores capacity");
}

void testExpertMovementAndEviction() {
    hypermoe::MemoryManager memory(8 * MiB, 16 * MiB);
    hypermoe::ExpertManager manager(
        memory, std::make_unique<hypermoe::LruCachePolicy>());

    for (hypermoe::ExpertId id = 0; id < 4; ++id) {
        manager.registerExpert({id, id / 2, 4 * MiB,
                                hypermoe::QuantizationType::Q4,
                                hypermoe::MemoryTier::Nvme});
    }

    expect(manager.requestExpert(0).source == hypermoe::RequestSource::NvmeLoad,
           "first request loads from NVMe");
    expect(manager.requestExpert(0).source == hypermoe::RequestSource::VramHit,
           "repeat request hits VRAM");
    (void)manager.requestExpert(1);
    (void)manager.requestExpert(2);
    expect(manager.requestExpert(0).source == hypermoe::RequestSource::RamHit,
           "demoted expert is served from the warm tier");

    const auto snapshot = memory.snapshot();
    expect(snapshot.vram.usedBytes <= snapshot.vram.limitBytes, "VRAM remains bounded");
    expect(snapshot.ram.usedBytes <= snapshot.ram.limitBytes, "RAM remains bounded");
    expect(manager.findExpert(2)->location == hypermoe::MemoryTier::Vram,
           "requested expert ends in VRAM");
    expect(manager.stats().vramEvictions >= 1, "capacity pressure evicts an expert");
}

void testOversizedExpertFailsWithoutLeak() {
    hypermoe::MemoryManager memory(2 * MiB, 2 * MiB);
    hypermoe::ExpertManager manager(
        memory, std::make_unique<hypermoe::LruCachePolicy>());
    manager.registerExpert({7, 0, 3 * MiB, hypermoe::QuantizationType::Q4,
                            hypermoe::MemoryTier::Nvme});
    try {
        (void)manager.requestExpert(7);
        expect(false, "oversized expert request throws");
    } catch (const std::runtime_error&) {
        expect(true, "oversized expert request throws");
    }
    const auto snapshot = memory.snapshot();
    expect(snapshot.vram.usedBytes == 0 && snapshot.ram.usedBytes == 0,
           "failed movement does not leak reservations");
}

void testFailedInitialResidencyDoesNotLeak() {
    hypermoe::MemoryManager memory(2 * MiB, 8 * MiB);
    hypermoe::ExpertManager manager(
        memory, std::make_unique<hypermoe::LruCachePolicy>());
    try {
        manager.registerExpert({8, 0, 4 * MiB, hypermoe::QuantizationType::Q4,
                                hypermoe::MemoryTier::Vram});
        expect(false, "invalid initial VRAM residency throws");
    } catch (const std::runtime_error&) {
        expect(true, "invalid initial VRAM residency throws");
    }
    const auto snapshot = memory.snapshot();
    expect(snapshot.vram.usedBytes == 0 && snapshot.ram.usedBytes == 0,
           "failed registration releases intermediate RAM reservation");
    expect(manager.expertCount() == 0, "failed registration removes expert ownership");
}

} // namespace

int main() {
    testMemoryLimitsAndRelease();
    testExpertMovementAndEviction();
    testOversizedExpertFailsWithoutLeak();
    testFailedInitialResidencyDoesNotLeak();
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 1 tests passed\n";
    return EXIT_SUCCESS;
}
