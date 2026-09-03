#pragma once

#include "backend/cuda/CudaDeviceInfo.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace hypermoe::backend {

enum class CudaValidationStatus {
    Passed,
    Skipped,
    Failed,
};

[[nodiscard]] std::string_view toString(CudaValidationStatus status) noexcept;

struct CudaValidationCheck {
    std::string name;
    bool passed{};
    std::string detail;
};

struct CudaRuntimeValidationReport {
    CudaValidationStatus status{CudaValidationStatus::Skipped};
    bool compiledWithCuda{};
    bool runtimeAvailable{};
    CudaDeviceInfo device;
    std::string message;
    std::vector<CudaValidationCheck> checks;

    [[nodiscard]] bool passed() const noexcept;
    [[nodiscard]] bool skipped() const noexcept;
    [[nodiscard]] std::string toJson() const;
    void writeJson(const std::filesystem::path& path) const;
};

class CudaRuntimeValidator {
public:
    [[nodiscard]] static CudaRuntimeValidationReport validate(int device = 0) noexcept;
    [[nodiscard]] static CudaRuntimeValidationReport validateAndWrite(
        const std::filesystem::path& reportPath,
        int device = 0);
};

} // namespace hypermoe::backend
