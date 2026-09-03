#pragma once

#include <cstdint>
#include <string>

namespace hypermoe::backend {

struct CudaDeviceInfo {
    bool available{};
    int deviceOrdinal{};
    std::string name;
    int computeCapabilityMajor{};
    int computeCapabilityMinor{};
    std::uint64_t totalVramBytes{};
    std::uint64_t freeVramBytes{};
    std::uint32_t streamCount{};
    int runtimeVersion{};
    int driverVersion{};

    [[nodiscard]] std::string computeCapability() const;
    [[nodiscard]] std::string toJson() const;
};

// Preserve the Phase 4 API while giving the CUDA-specific record an explicit name.
using DeviceInfo = CudaDeviceInfo;

} // namespace hypermoe::backend
