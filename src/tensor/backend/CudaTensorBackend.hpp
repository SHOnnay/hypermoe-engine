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
    void copyTensor(const Tensor& source, Tensor& destination) override;
    void matmul(const Tensor& left,
                const Tensor& right,
                Tensor& output) override;
    void add(const Tensor& left,
             const Tensor& right,
             Tensor& output) override;
    void mul(const Tensor& left,
             const Tensor& right,
             Tensor& output) override;
    [[nodiscard]] Tensor reshape(const Tensor& tensor, Shape shape) override;
    void synchronize() override;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace hypermoe::tensor
