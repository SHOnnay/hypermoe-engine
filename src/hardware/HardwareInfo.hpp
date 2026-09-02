#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace hypermoe::hardware {

struct HardwareInfo {
    std::string operatingSystem;
    std::string cpuName;
    std::uint32_t logicalCpuCount{};
    std::uint64_t ramBytes{};
    std::uint64_t availableStorageBytes{};
    bool cudaCompiled{};
    bool cudaAvailable{};
    std::string gpuName;
    std::uint64_t vramBytes{};
    int cudaRuntimeVersion{};
    int cudaDriverVersion{};

    [[nodiscard]] std::string toJson() const;
};

[[nodiscard]] HardwareInfo
detectHardware(const std::filesystem::path& storagePath = std::filesystem::current_path());

} // namespace hypermoe::hardware
