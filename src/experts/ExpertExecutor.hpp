#pragma once

#include "tensor/TensorView.hpp"
#include "tensor/activation/Activation.hpp"
#include "tensor/quantization/Quantization.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe {
class Profiler;

struct ExpertMlpWeights {
    ExpertMlpWeights() = default;
    ExpertMlpWeights(
        tensor::TensorView gate,
        tensor::TensorView up,
        tensor::TensorView down,
        std::optional<tensor::quantization::QuantizationParameters>
            gateParameters = {},
        std::optional<tensor::quantization::QuantizationParameters>
            upParameters = {},
        std::optional<tensor::quantization::QuantizationParameters>
            downParameters = {})
        : gateProjection(std::move(gate)),
          upProjection(std::move(up)),
          downProjection(std::move(down)),
          gateQuantization(gateParameters),
          upQuantization(upParameters),
          downQuantization(downParameters) {}

    tensor::TensorView gateProjection;
    tensor::TensorView upProjection;
    tensor::TensorView downProjection;
    std::optional<tensor::quantization::QuantizationParameters> gateQuantization;
    std::optional<tensor::quantization::QuantizationParameters> upQuantization;
    std::optional<tensor::quantization::QuantizationParameters> downQuantization;
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
