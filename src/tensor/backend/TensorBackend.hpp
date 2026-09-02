#pragma once

#include "tensor/Tensor.hpp"

#include <string_view>

namespace hypermoe::tensor {

class TensorBackend {
public:
    virtual ~TensorBackend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual Device device() const noexcept = 0;
    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual Tensor allocateTensor(const Shape& shape, DType dtype) = 0;
    virtual void copyTensor(const Tensor& source, Tensor& destination) = 0;
    virtual void matmul(const Tensor& left,
                        const Tensor& right,
                        Tensor& output) = 0;
    virtual void add(const Tensor& left,
                     const Tensor& right,
                     Tensor& output) = 0;
    virtual void mul(const Tensor& left,
                     const Tensor& right,
                     Tensor& output) = 0;
    [[nodiscard]] virtual Tensor reshape(const Tensor& tensor, Shape shape) = 0;
    virtual void synchronize() = 0;
};

} // namespace hypermoe::tensor
