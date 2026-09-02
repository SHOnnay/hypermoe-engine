#pragma once

#include "backend/Backend.hpp"
#include "tensor/DType.hpp"
#include "tensor/Shape.hpp"

#include <cstddef>
#include <memory>
#include <string_view>

namespace hypermoe::tensor {

enum class DeviceType {
    CPU,
    CUDA,
};

[[nodiscard]] constexpr std::string_view toString(DeviceType type) noexcept {
    switch (type) {
    case DeviceType::CPU: return "CPU";
    case DeviceType::CUDA: return "CUDA";
    }
    return "UNKNOWN";
}

struct Device {
    DeviceType type{DeviceType::CPU};
    int ordinal{};

    [[nodiscard]] static constexpr Device cpu() noexcept { return {}; }
    [[nodiscard]] static constexpr Device cuda(int ordinal = 0) noexcept {
        return {DeviceType::CUDA, ordinal};
    }
    [[nodiscard]] constexpr bool operator==(const Device&) const noexcept = default;
};

class Tensor {
public:
    Tensor() = default;

    [[nodiscard]] static Tensor fromStorage(Shape shape,
                                            DType dtype,
                                            Device device,
                                            void* data,
                                            std::size_t storageBytes,
                                            std::shared_ptr<void> owner);
    [[nodiscard]] static Tensor
    fromDeviceBuffer(Shape shape,
                     DType dtype,
                     Device device,
                     std::shared_ptr<backend::DeviceBuffer> buffer);

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] std::size_t storageBytes() const noexcept;
    [[nodiscard]] bool isContiguous() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] Tensor reshape(Shape shape) const;

private:
    Tensor(Shape shape,
           DType dtype,
           Device device,
           void* data,
           std::size_t storageBytes,
           std::shared_ptr<void> owner);
    [[nodiscard]] static std::size_t requiredBytes(const Shape& shape, DType dtype);
    [[nodiscard]] static std::size_t logicalBytes(const Shape& shape, DType dtype);

    Shape shape_;
    DType dtype_{DType::FP32};
    Device device_;
    void* data_{};
    std::size_t bytes_{};
    std::size_t storageBytes_{};
    std::shared_ptr<void> owner_;
};

} // namespace hypermoe::tensor
