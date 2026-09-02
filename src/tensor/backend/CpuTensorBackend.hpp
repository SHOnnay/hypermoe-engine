#pragma once

#include "tensor/backend/TensorBackend.hpp"

#include <memory>

namespace hypermoe {
class Profiler;
}

namespace hypermoe::tensor {

class CpuTensorBackend final : public TensorBackend {
public:
    explicit CpuTensorBackend(std::shared_ptr<Profiler> profiler = {});

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
    std::shared_ptr<backend::ComputeBackend> backend_;
    std::shared_ptr<Profiler> profiler_;
};

} // namespace hypermoe::tensor
