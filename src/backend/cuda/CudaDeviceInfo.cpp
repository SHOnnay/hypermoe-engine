#include "backend/cuda/CudaDeviceInfo.hpp"

#include <sstream>

namespace hypermoe::backend {
namespace {

std::string jsonEscape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

} // namespace

std::string CudaDeviceInfo::computeCapability() const {
    if (!available) return {};
    return std::to_string(computeCapabilityMajor) + "." +
           std::to_string(computeCapabilityMinor);
}

std::string CudaDeviceInfo::toJson() const {
    std::ostringstream output;
    output << "{\n"
           << "  \"available\": " << (available ? "true" : "false") << ",\n"
           << "  \"device_ordinal\": " << deviceOrdinal << ",\n"
           << "  \"name\": \"" << jsonEscape(name) << "\",\n"
           << "  \"compute_capability\": \"" << computeCapability() << "\",\n"
           << "  \"total_vram_bytes\": " << totalVramBytes << ",\n"
           << "  \"free_vram_bytes\": " << freeVramBytes << ",\n"
           << "  \"stream_count\": " << streamCount << ",\n"
           << "  \"cuda_runtime_version\": " << runtimeVersion << ",\n"
           << "  \"cuda_driver_version\": " << driverVersion << "\n"
           << "}";
    return output.str();
}

} // namespace hypermoe::backend
