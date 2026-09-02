#include "experts/ExpertExecutor.hpp"

#include "profiling/Profiler.hpp"
#include "tensor/backend/TensorBackend.hpp"

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
    backend_->matmul(input, expertWeights, output);
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
    const auto validTensor = [&](tensor::TensorView view) {
        return view && view.device() == expectedDevice && view.isContiguous() &&
               view.dtype() == tensor::DType::FP32 && view.shape().rank() == 2;
    };
    if (!validTensor(input) || !validTensor(weights.gateProjection) ||
        !validTensor(weights.upProjection) ||
        !validTensor(weights.downProjection) || !validTensor(output) ||
        !output.writable()) {
        throw std::invalid_argument(
            "expert MLP requires contiguous rank-2 FP32 tensors on one backend");
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
    const tensor::Shape intermediateShape{inputShape[0], gateShape[1]};
    auto gate = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);
    auto up = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);
    auto activated = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);
    auto gated = backend_->allocateTensor(intermediateShape, tensor::DType::FP32);

    std::chrono::steady_clock::duration projectionTime{};
    auto projectionStart = std::chrono::steady_clock::now();
    backend_->matmul(input, weights.gateProjection, gate);
    projectionTime += std::chrono::steady_clock::now() - projectionStart;

    projectionStart = std::chrono::steady_clock::now();
    backend_->matmul(input, weights.upProjection, up);
    projectionTime += std::chrono::steady_clock::now() - projectionStart;

    tensor::activation::apply(activation_, *backend_, gate, activated, profiler_);
    backend_->mul(activated, up, gated);

    projectionStart = std::chrono::steady_clock::now();
    backend_->matmul(gated, weights.downProjection, output);
    projectionTime += std::chrono::steady_clock::now() - projectionStart;
    backend_->synchronize();

    if (profiler_) {
        profiler_->recordProjectionTime(projectionTime);
        profiler_->recordExpertExecutionTime(
            std::chrono::steady_clock::now() - expertStart);
    }
}

} // namespace hypermoe
