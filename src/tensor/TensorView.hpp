#pragma once

#include "tensor/Tensor.hpp"

#include <cstddef>
#include <memory>

namespace hypermoe::tensor {

// TensorView never extends storage lifetime. Consumers must keep the originating
// Tensor/DeviceBuffer resident for the complete operation.
class TensorView {
public:
    TensorView() = default;
    TensorView(Tensor& tensor);
    TensorView(const Tensor& tensor);
    TensorView(Tensor&&) = delete;

    [[nodiscard]] static TensorView
    fromDeviceBuffer(const Shape& shape,
                     DType dtype,
                     Device device,
                     const std::shared_ptr<backend::DeviceBuffer>& buffer,
                     bool writable = true);

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] void* mutableData() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] std::size_t storageBytes() const noexcept;
    [[nodiscard]] bool isContiguous() const noexcept;
    [[nodiscard]] bool writable() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::shared_ptr<void> lockOwner() const noexcept;

    [[nodiscard]] TensorView reshape(Shape shape) const;
    [[nodiscard]] TensorView sliceBytes(std::size_t offsetBytes,
                                        Shape shape,
                                        DType dtype) const;

private:
    TensorView(Shape shape,
               DType dtype,
               Device device,
               const void* data,
               void* mutableData,
               std::size_t storageBytes,
               std::weak_ptr<void> lifetime);

    [[nodiscard]] static std::size_t requiredBytes(const Shape& shape, DType dtype);
    [[nodiscard]] static std::size_t logicalBytes(const Shape& shape, DType dtype);

    Shape shape_;
    DType dtype_{DType::FP32};
    Device device_;
    const void* data_{};
    void* mutableData_{};
    std::size_t bytes_{};
    std::size_t storageBytes_{};
    std::weak_ptr<void> lifetime_;
};

} // namespace hypermoe::tensor
