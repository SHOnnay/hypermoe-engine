#pragma once

#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"

#include <cstddef>
#include <cstdint>

namespace hypermoe {

struct MemoryPressureConfig {
    std::size_t vramSafetyMarginBytes{};
    std::size_t ramSafetyMarginBytes{};
    std::size_t storageQueueHighWatermark{64};
};

struct MemoryPressureReport {
    bool vramPressure{};
    bool ramPressure{};
    bool storageQueuePressure{};
    std::size_t vramExpertsMoved{};
    std::size_t ramExpertsMoved{};
    std::size_t storageQueueDepth{};
};

struct MemoryPressureStats {
    std::uint64_t polls{};
    std::uint64_t vramPressureEvents{};
    std::uint64_t ramPressureEvents{};
    std::uint64_t storageQueuePressureEvents{};
};

class MemoryPressureController {
public:
    MemoryPressureController(MemoryManager& memory,
                             ExpertManager& experts,
                             const TransferManager& transfers,
                             MemoryPressureConfig config);

    [[nodiscard]] MemoryPressureReport poll();
    [[nodiscard]] MemoryPressureStats stats() const noexcept;

private:
    MemoryManager& memory_;
    ExpertManager& experts_;
    const TransferManager& transfers_;
    MemoryPressureConfig config_;
    MemoryPressureStats stats_;
};

} // namespace hypermoe
