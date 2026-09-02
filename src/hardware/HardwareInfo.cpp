#include "hardware/HardwareInfo.hpp"

#include "backend/CudaBackend.hpp"

#include <sstream>
#include <thread>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <fstream>
#endif

namespace hypermoe::hardware {
namespace {

std::string jsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

#if defined(__APPLE__)
template <typename T>
T sysctlValue(const char* name) {
    T value{};
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : T{};
}

std::string sysctlString(const char* name) {
    std::size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return {};
    std::string result(size, '\0');
    if (sysctlbyname(name, result.data(), &size, nullptr, 0) != 0) return {};
    while (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
}
#endif

} // namespace

std::string HardwareInfo::toJson() const {
    std::ostringstream output;
    output << "{\n"
           << "  \"operating_system\": \"" << jsonEscape(operatingSystem) << "\",\n"
           << "  \"cpu_name\": \"" << jsonEscape(cpuName) << "\",\n"
           << "  \"logical_cpu_count\": " << logicalCpuCount << ",\n"
           << "  \"ram_bytes\": " << ramBytes << ",\n"
           << "  \"available_storage_bytes\": " << availableStorageBytes << ",\n"
           << "  \"cuda_compiled\": " << (cudaCompiled ? "true" : "false") << ",\n"
           << "  \"cuda_available\": " << (cudaAvailable ? "true" : "false") << ",\n"
           << "  \"gpu_name\": \"" << jsonEscape(gpuName) << "\",\n"
           << "  \"compute_capability\": \"" << jsonEscape(computeCapability)
           << "\",\n"
           << "  \"vram_bytes\": " << vramBytes << ",\n"
           << "  \"free_vram_bytes\": " << freeVramBytes << ",\n"
           << "  \"cuda_runtime_version\": " << cudaRuntimeVersion << ",\n"
           << "  \"cuda_driver_version\": " << cudaDriverVersion << "\n"
           << "}";
    return output.str();
}

HardwareInfo detectHardware(const std::filesystem::path& storagePath) {
    HardwareInfo info;
    info.logicalCpuCount = std::thread::hardware_concurrency();
#if defined(_WIN32)
    info.operatingSystem = "Windows";
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) info.ramBytes = memory.ullTotalPhys;
    info.cpuName = "Windows CPU";
#elif defined(__APPLE__)
    info.operatingSystem = "macOS";
    info.ramBytes = sysctlValue<std::uint64_t>("hw.memsize");
    info.cpuName = sysctlString("machdep.cpu.brand_string");
    if (info.cpuName.empty()) info.cpuName = sysctlString("hw.model");
#elif defined(__linux__)
    info.operatingSystem = "Linux";
    struct sysinfo memory {};
    if (sysinfo(&memory) == 0) {
        info.ramBytes = static_cast<std::uint64_t>(memory.totalram) * memory.mem_unit;
    }
    std::ifstream cpuInfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuInfo, line)) {
        constexpr std::string_view marker = "model name";
        if (line.starts_with(marker)) {
            const auto separator = line.find(':');
            if (separator != std::string::npos) info.cpuName = line.substr(separator + 2);
            break;
        }
    }
#else
    info.operatingSystem = "Unknown";
    info.cpuName = "Unknown CPU";
#endif
    std::error_code storageError;
    const auto space = std::filesystem::space(storagePath, storageError);
    if (!storageError) info.availableStorageBytes = space.available;

    info.cudaCompiled = backend::CudaBackend::compiledWithCuda();
    const auto cuda = backend::CudaBackend::query();
    info.cudaAvailable = cuda.available;
    info.gpuName = cuda.deviceName;
    if (cuda.available) {
        info.computeCapability = std::to_string(cuda.computeCapabilityMajor) + "." +
                                 std::to_string(cuda.computeCapabilityMinor);
    }
    info.vramBytes = cuda.totalVramBytes;
    info.freeVramBytes = cuda.freeVramBytes;
    info.cudaRuntimeVersion = cuda.runtimeVersion;
    info.cudaDriverVersion = cuda.driverVersion;
    return info;
}

} // namespace hypermoe::hardware
