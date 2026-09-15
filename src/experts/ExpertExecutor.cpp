#include "experts/ExpertExecutor.hpp"

#include "profiling/Profiler.hpp"
#include "tensor/backend/TensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace hypermoe {

MatmulExpertExecutor::MatmulExpertExecutor(
    std::shared_ptr<tensor::TensorBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available()) {
        throw std::invalid_argument("expert executor requires an available tensor backend");
    }
}

void MatmulExpertExecutor::execute(tensor::TensorView input,
                                   tensor::TensorView expertWeights,
                                   tensor::TensorView output) {
    if (auto* cuda = dynamic_cast<tensor::CudaTensorBackend*>(backend_.get())) {
        cuda->matmulExpert(input, expertWeights, output);
    } else {
        backend_->matmul(input, expertWeights, output);
    }
}

ExpertMlpExecutor::ExpertMlpExecutor(
    std::shared_ptr<tensor::TensorBackend> backend,
    tensor::activation::ActivationType activation,
    std::shared_ptr<Profiler> profiler)
    : backend_(std::move(backend)),
      activation_(activation),
      profiler_(std::move(profiler)) {
    if (!backend_ || !backend_->available()) {
        throw std::invalid_argument("expert MLP requires an available tensor backend");
    }
}

void ExpertMlpExecutor::execute(tensor::TensorView input,
                                const ExpertMlpWeights& weights,
                                tensor::TensorView output) {
    [[maybe_unused]] const auto inputOwner = input.lockOwner();
    [[maybe_unused]] const auto gateOwner = weights.gateProjection.lockOwner();
    [[maybe_unused]] const auto upOwner = weights.upProjection.lockOwner();
    [[maybe_unused]] const auto downOwner = weights.downProjection.lockOwner();
    [[maybe_unused]] const auto outputOwner = output.lockOwner();
    if (!inputOwner || !gateOwner || !upOwner || !downOwner || !outputOwner) {
        throw std::invalid_argument("expert MLP received expired tensor storage");
    }
    const auto expectedDevice = backend_->device();
    const auto validActivation = [&](tensor::TensorView view) {
        return view && view.device() == expectedDevice && view.isContiguous() &&
               view.dtype() == tensor::DType::FP32 && view.shape().rank() == 2;
    };
    const auto validWeight = [&](
        tensor::TensorView view,
        const std::optional<tensor::quantization::QuantizationParameters>&
            quantization) {
        if (!view || view.device() != expectedDevice || !view.isContiguous() ||
            view.shape().rank() != 2) {
            return false;
        }
        if (view.dtype() == tensor::DType::FP32) return !quantization.has_value();
        if (view.dtype() != tensor::DType::INT8 || !quantization) return false;
        tensor::quantization::validateParameters(
            tensor::quantization::QuantizedDType::INT8, *quantization);
        return true;
    };
    if (!validActivation(input) ||
        !validWeight(weights.gateProjection, weights.gateQuantization) ||
        !validWeight(weights.upProjection, weights.upQuantization) ||
        !validWeight(weights.downProjection, weights.downQuantization) ||
        !validActivation(output) || !output.writable()) {
        throw std::invalid_argument(
            "expert MLP requires FP32 activations and FP32 or metadata-backed INT8 weights on one backend");
    }

    const auto& inputShape = input.shape().dimensions();
    const auto& gateShape = weights.gateProjection.shape().dimensions();
    const auto& upShape = weights.upProjection.shape().dimensions();
    const auto& downShape = weights.downProjection.shape().dimensions();
    const auto& outputShape = output.shape().dimensions();
    if (gateShape != upShape || inputShape[1] != gateShape[0] ||
        gateShape[1] != downShape[0] || inputShape[0] != outputShape[0] ||
        downShape[1] != outputShape[1]) {
        throw std::invalid_argument("expert MLP projection dimensions are incompatible");
    }

    const auto expertStart = std::chrono::steady_clock::now();
    auto* cuda = dynamic_cast<tensor::CudaTensorBackend*>(backend_.get());
    std::optional<profiling::GpuEventQueue::Scope> gpuRegion;
    if (cuda) gpuRegion.emplace(cuda->timeRegion(profiling::GpuOperation::ExpertRegion));
    const tensor::Shape intermediateShape{inputShape[0], gateShape[1]};
    auto gate = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);
    auto up = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);
    auto activated = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);
    auto gated = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);

    std::chrono::steady_clock::duration projectionTime{};
    const auto project = [&](tensor::TensorView source,
                             tensor::TensorView weight,
                             const std::optional<
                                 tensor::quantization::QuantizationParameters>&
                                 quantization,
                             tensor::TensorView destination) {
        if (weight.dtype() == tensor::DType::INT8) {
            if (cuda) cuda->matmulInt8Expert(source, weight, *quantization, destination);
            else backend_->matmulInt8Weights(source, weight, *quantization, destination);
        } else {
            if (cuda) cuda->matmulExpert(source, weight, destination);
            else backend_->matmul(source, weight, destination);
        }
    };
    auto projectionStart = std::chrono::steady_clock::now();
    project(input, weights.gateProjection, weights.gateQuantization, gate);
    projectionTime += std::chrono::steady_clock::now() - projectionStart;

    projectionStart = std::chrono::steady_clock::now();
    project(input, weights.upProjection, weights.upQuantization, up);
    projectionTime += std::chrono::steady_clock::now() - projectionStart;

    tensor::activation::apply(activation_, *backend_, gate, activated, profiler_);
    backend_->mul(activated, up, gated);

    projectionStart = std::chrono::steady_clock::now();
    project(gated, weights.downProjection, weights.downQuantization, output);
    projectionTime += std::chrono::steady_clock::now() - projectionStart;
    if (gpuRegion) gpuRegion->finish();
    if (profiler_ && !cuda) {
        profiler_->recordProjectionTime(projectionTime);
        profiler_->recordExpertExecutionTime(
            std::chrono::steady_clock::now() - expertStart);
    }
}

} // namespace hypermoe
