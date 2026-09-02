#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/quantization/Quantization.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace hypermoe::tensor::quantization {

struct QuantizedTensorMetadata {
    std::uint32_t version{1};
    QuantizedDType dtype{QuantizedDType::INT8};
    QuantizationParameters parameters;
    Shape shape;
    Device device;
    std::size_t storageBytes{};

    [[nodiscard]] std::string toJson() const;
};

class QuantizedTensor {
public:
    QuantizedTensor() = default;

    [[nodiscard]] static QuantizedTensor
    fromStorage(Shape shape,
                QuantizedDType dtype,
                QuantizationParameters parameters,
                Device device,
                void* data,
                std::size_t storageBytes,
                std::shared_ptr<void> owner);
    [[nodiscard]] static QuantizedTensor
    fromDeviceBuffer(Shape shape,
                     QuantizedDType dtype,
                     QuantizationParameters parameters,
                     Device device,
                     std::shared_ptr<backend::DeviceBuffer> buffer);

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] QuantizedDType dtype() const noexcept;
    [[nodiscard]] const QuantizationParameters& parameters() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] std::size_t storageBytes() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] QuantizedTensorMetadata metadata() const;

private:
    QuantizedTensor(Shape shape,
                    QuantizedDType dtype,
                    QuantizationParameters parameters,
                    Device device,
                    void* data,
                    std::size_t storageBytes,
                    std::shared_ptr<void> owner);

    Shape shape_;
    QuantizedDType dtype_{QuantizedDType::INT8};
    QuantizationParameters parameters_;
    Device device_;
    void* data_{};
    std::size_t bytes_{};
    std::size_t storageBytes_{};
    std::shared_ptr<void> owner_;
};

} // namespace hypermoe::tensor::quantization
