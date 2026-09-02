#include "memory/MemoryPressureController.hpp"

namespace hypermoe {

MemoryPressureController::MemoryPressureController(
    MemoryManager& memory,
    ExpertManager& experts,
    const TransferManager& transfers,
    MemoryPressureConfig config)
    : memory_(memory), experts_(experts), transfers_(transfers), config_(config) {}

MemoryPressureReport MemoryPressureController::poll() {
    ++stats_.polls;
    MemoryPressureReport report;
    const auto initial = memory_.snapshot();
    const auto vramTarget = initial.vram.limitBytes > config_.vramSafetyMarginBytes
                                ? initial.vram.limitBytes - config_.vramSafetyMarginBytes
                                : 0;
    const auto ramTarget = initial.ram.limitBytes > config_.ramSafetyMarginBytes
                               ? initial.ram.limitBytes - config_.ramSafetyMarginBytes
                               : 0;

    report.vramPressure = initial.vram.usedBytes > vramTarget;
    if (report.vramPressure) {
        ++stats_.vramPressureEvents;
        report.vramExpertsMoved = experts_.evictUntilWithin(MemoryTier::Vram, vramTarget);
    }

    // VRAM demotions can increase RAM use, so sample again before warm eviction.
    report.ramPressure = memory_.snapshot().ram.usedBytes > ramTarget;
    if (report.ramPressure) {
        ++stats_.ramPressureEvents;
        report.ramExpertsMoved = experts_.evictUntilWithin(MemoryTier::Ram, ramTarget);
    }

    report.storageQueueDepth = transfers_.pending();
    report.storageQueuePressure =
        report.storageQueueDepth >= config_.storageQueueHighWatermark;
    if (report.storageQueuePressure) {
        ++stats_.storageQueuePressureEvents;
    }
    return report;
}

MemoryPressureStats MemoryPressureController::stats() const noexcept {
    return stats_;
}

} // namespace hypermoe
