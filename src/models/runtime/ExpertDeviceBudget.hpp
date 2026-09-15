#pragma once

#include "backend/Backend.hpp"
#include "backend/cuda/CudaMemoryPool.hpp"

#include <cstddef>

namespace hypermoe::models::runtime {

// Reservations are envelopes, not allocations. They apply only in automatic
// mode; manual mode preserves the Phase 22B capacity and preflight semantics.
struct ExpertDeviceReservations {
    std::size_t kvCacheBytes{256ULL * 1024ULL * 1024ULL};
    std::size_t workspaceBytes{512ULL * 1024ULL * 1024ULL};
    std::size_t stagingPoolBytes{backend::CudaMemoryPool::defaultMaximumCachedBytes};
    std::size_t safetyBytes{512ULL * 1024ULL * 1024ULL};
};

struct ExpertDeviceBudgetPlan {
    bool automatic{};
    std::size_t configuredBytes{};
    std::size_t effectiveBytes{};
    std::size_t totalDeviceBytes{};
    std::size_t freeDeviceBytesAfterStatic{};
    std::size_t staticTensorBytes{};
    ExpertDeviceReservations reservations;
    std::size_t transferAllowanceBytes{};
    std::size_t reservedBytes{};
};

// Pure arithmetic, testable without CUDA. freeBytes must be sampled AFTER
// static tensors/context initialization, so static tensors are not subtracted
// from freeBytes a second time. totalBytes provides a second conservative bound.
[[nodiscard]] ExpertDeviceBudgetPlan planExpertDeviceBudget(
    std::size_t configuredBytes, bool automatic,
    backend::MemoryInfo memory, std::size_t staticTensorBytes,
    ExpertDeviceReservations reservations, std::size_t largestExpertBytes,
    std::size_t transferAllowanceBytes = 0);

} // namespace hypermoe::models::runtime
