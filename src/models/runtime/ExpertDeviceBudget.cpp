#include "models/runtime/ExpertDeviceBudget.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hypermoe::models::runtime {
namespace {
std::size_t checkedAdd(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error("expert device reservation accounting overflows");
    }
    return left + right;
}
} // namespace

ExpertDeviceBudgetPlan planExpertDeviceBudget(
    std::size_t configuredBytes, bool automatic, backend::MemoryInfo memory,
    std::size_t staticTensorBytes, ExpertDeviceReservations reservations,
    std::size_t largestExpertBytes, std::size_t transferAllowanceBytes) {
    if (configuredBytes == 0 || largestExpertBytes == 0 || memory.totalBytes == 0 ||
        memory.freeBytes > memory.totalBytes || staticTensorBytes > memory.totalBytes ||
        memory.totalBytes > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid GPU memory snapshot or expert device budget");
    }
    ExpertDeviceBudgetPlan result;
    result.automatic = automatic;
    result.configuredBytes = configuredBytes;
    result.totalDeviceBytes = static_cast<std::size_t>(memory.totalBytes);
    result.freeDeviceBytesAfterStatic = static_cast<std::size_t>(memory.freeBytes);
    result.staticTensorBytes = staticTensorBytes;
    result.reservations = reservations;
    result.transferAllowanceBytes = transferAllowanceBytes;
    const auto available = std::min(result.freeDeviceBytesAfterStatic,
                                   result.totalDeviceBytes - staticTensorBytes);
    if (automatic) {
        if (reservations.workspaceBytes == 0 || reservations.safetyBytes == 0 ||
            reservations.stagingPoolBytes < backend::CudaMemoryPool::defaultMaximumCachedBytes) {
            throw std::invalid_argument("automatic expert budget requires workspace, safety, and the 512MiB transfer pool reservation");
        }
        result.reservedBytes = checkedAdd(
            checkedAdd(reservations.kvCacheBytes, reservations.workspaceBytes),
            checkedAdd(checkedAdd(reservations.stagingPoolBytes, reservations.safetyBytes),
                       transferAllowanceBytes));
        if (result.reservedBytes >= available) {
            throw std::invalid_argument("GPU headroom reservations leave no memory for experts");
        }
        // Match the existing pool's alignment, rounding DOWN, never up.
        result.effectiveBytes = (available - result.reservedBytes) / 256U * 256U;
    } else {
        if (configuredBytes > available) {
            throw std::invalid_argument("expert VRAM budget exceeds free device memory after static tensor loading");
        }
        result.effectiveBytes = configuredBytes;
    }
    if (result.effectiveBytes < largestExpertBytes) {
        throw std::invalid_argument("safe expert device budget cannot hold the largest indexed expert");
    }
    return result;
}
} // namespace hypermoe::models::runtime
