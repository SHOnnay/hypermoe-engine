#pragma once

#include "tensor/Tensor.hpp"

#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe {

class ExpertExecutor {
public:
    virtual ~ExpertExecutor() = default;
    virtual void execute(const tensor::Tensor& input,
                         const tensor::Tensor& expertWeights,
                         tensor::Tensor& output) = 0;
};

class MatmulExpertExecutor final : public ExpertExecutor {
public:
    explicit MatmulExpertExecutor(std::shared_ptr<tensor::TensorBackend> backend);

    void execute(const tensor::Tensor& input,
                 const tensor::Tensor& expertWeights,
                 tensor::Tensor& output) override;

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
};

} // namespace hypermoe
