#pragma once

#include "tensor/TensorView.hpp"
#include "tensor/activation/Activation.hpp"

#include <memory>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe {
class Profiler;

struct ExpertMlpWeights {
    tensor::TensorView gateProjection;
    tensor::TensorView upProjection;
    tensor::TensorView downProjection;
};

class ExpertExecutor {
public:
    virtual ~ExpertExecutor() = default;
    virtual void execute(tensor::TensorView input,
                         tensor::TensorView expertWeights,
                         tensor::TensorView output) = 0;
};

class MatmulExpertExecutor final : public ExpertExecutor {
public:
    explicit MatmulExpertExecutor(std::shared_ptr<tensor::TensorBackend> backend);

    void execute(tensor::TensorView input,
                 tensor::TensorView expertWeights,
                 tensor::TensorView output) override;

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
};

// Executes the gated expert primitive used by common sparse MoE families:
// down(activation(input * gate) elementwise-mul (input * up)).
class ExpertMlpExecutor final {
public:
    explicit ExpertMlpExecutor(
        std::shared_ptr<tensor::TensorBackend> backend,
        tensor::activation::ActivationType activation =
            tensor::activation::ActivationType::SiLU,
        std::shared_ptr<Profiler> profiler = {});

    void execute(tensor::TensorView input,
                 const ExpertMlpWeights& weights,
                 tensor::TensorView output);

private:
    std::shared_ptr<tensor::TensorBackend> backend_;
    tensor::activation::ActivationType activation_;
    std::shared_ptr<Profiler> profiler_;
};

} // namespace hypermoe
