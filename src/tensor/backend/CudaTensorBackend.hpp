#pragma once

#include "tensor/backend/TensorBackend.hpp"

#include <memory>

namespace hypermoe {
class Profiler;
}

namespace hypermoe::tensor {

class CudaTensorBackend final : public TensorBackend {
public:
    struct Impl;

    explicit CudaTensorBackend(int device = 0,
                               std::shared_ptr<Profiler> profiler = {});
    ~CudaTensorBackend() override;

    CudaTensorBackend(const CudaTensorBackend&) = delete;
    CudaTensorBackend& operator=(const CudaTensorBackend&) = delete;

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] Device device() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] Tensor allocateTensor(const Shape& shape, DType dtype) override;
    void copyTensor(TensorView source, TensorView destination) override;
    void matmul(TensorView left,
                TensorView right,
                TensorView output) override;
    void add(TensorView left,
             TensorView right,
             TensorView output) override;
    void mul(TensorView left,
             TensorView right,
             TensorView output) override;
    [[nodiscard]] Tensor reshape(const Tensor& tensor, Shape shape) override;
    void synchronize() override;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace hypermoe::tensor
