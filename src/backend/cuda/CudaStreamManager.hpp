#pragma once

#include "backend/cuda/CudaRuntime.hpp"

#include <array>
#include <memory>
#include <string_view>

namespace hypermoe::backend {

enum class CudaStreamRole {
    Compute,
    Transfer,
    Prefetch,
};

[[nodiscard]] std::string_view toString(CudaStreamRole role) noexcept;

class CudaStreamManager {
public:
    explicit CudaStreamManager(std::shared_ptr<CudaRuntime> runtime);
    ~CudaStreamManager();

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;

    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] StreamHandle stream(CudaStreamRole role) const noexcept;
    void synchronize(CudaStreamRole role);
    void synchronizeAll();
    void shutdown() noexcept;

private:
    [[nodiscard]] static constexpr std::size_t index(CudaStreamRole role) noexcept {
        return static_cast<std::size_t>(role);
    }

    std::shared_ptr<CudaRuntime> runtime_;
    std::array<StreamHandle, 3> streams_{};
};

} // namespace hypermoe::backend
