#include "core/runtime/MoERuntime.hpp"

#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace hypermoe::runtime {
namespace {

void fillAndCopy(tensor::TensorBackend& backend,
                 tensor::Tensor& destination,
                 float value) {
    tensor::CpuTensorBackend cpu;
    auto host = cpu.allocateTensor(destination.shape(), tensor::DType::FP32);
    auto* values = static_cast<float*>(host.data());
    std::fill(values, values + host.shape().elementCount(), value);
    backend.copyTensor(host, destination);
}

} // namespace

MoERuntime::MoERuntime(
    std::shared_ptr<router::Router> router,
    std::shared_ptr<scheduler::Scheduler> scheduler,
    ExpertManager& experts,
    models::ExpertWeightMap weightMap,
    std::shared_ptr<tensor::TensorBackend> tensorBackend,
    std::shared_ptr<ExpertMlpExecutor> executor,
    std::shared_ptr<prediction::ExpertHistory> history,
    std::shared_ptr<prediction::ExpertPredictor> predictor)
    : router_(std::move(router)),
      scheduler_(std::move(scheduler)),
      experts_(experts),
      weightMap_(std::move(weightMap)),
      tensorBackend_(std::move(tensorBackend)),
      executor_(std::move(executor)),
      history_(std::move(history)),
      predictor_(std::move(predictor)) {
    if (!router_ || !scheduler_ || !tensorBackend_ || !tensorBackend_->available() ||
        !executor_) {
        throw std::invalid_argument("MoE runtime dependencies must be available");
    }
}

LayerExecutionResult MoERuntime::executeLayer(
    LayerId layerId,
    tensor::TensorView hiddenState,
    tensor::TensorView routerWeights) {
    std::scoped_lock executionLock(executionMutex_);
    if (!hiddenState || hiddenState.shape().rank() != 2 ||
        hiddenState.dtype() != tensor::DType::FP32 ||
        hiddenState.device() != tensorBackend_->device()) {
        throw std::invalid_argument(
            "MoE runtime requires a rank-2 FP32 hidden state on its tensor backend");
    }
    auto decision = router_->route(layerId, hiddenState, routerWeights);
    if (!decision.valid()) throw std::runtime_error("router returned an invalid decision");
    if (predictor_ && history_) {
        (void)predictor_->observeAndPrefetch(decision, *history_, *scheduler_);
    } else {
        if (history_) history_->record(decision);
        if (predictor_) predictor_->observe(decision);
    }

    std::vector<scheduler::ScheduleHandle> handles;
    handles.reserve(decision.selectedExpertIds.size());
    for (const auto expertId : decision.selectedExpertIds) {
        const auto expert = experts_.findExpert(layerId, expertId);
        if (!expert) throw std::out_of_range("router selected an unregistered expert");
        scheduler::ScheduleRequest request;
        request.layerId = layerId;
        request.expertId = expertId;
        request.source = expert->location;
        request.destination = MemoryTier::Vram;
        request.priority = scheduler::TransferPriority::ActiveInference;
        handles.push_back(scheduler_->schedule(std::move(request)));
    }

    std::vector<std::uint64_t> payloadOffsets;
    payloadOffsets.reserve(handles.size());
    for (std::size_t index = 0; index < handles.size(); ++index) {
        const auto& scheduled = handles[index].future().get();
        const auto expertId = decision.selectedExpertIds[index];
        if (!scheduled.success) {
            throw std::runtime_error("expert scheduling failed: " + scheduled.error);
        }
        if (scheduled.transfer.deviceBuffer) {
            experts_.adoptDeviceWeights(layerId, expertId,
                                        scheduled.transfer.deviceBuffer);
            payloadOffsets.push_back(scheduled.transfer.record.offset);
            payloadOffsets_[key(layerId, expertId)] = scheduled.transfer.record.offset;
        } else {
            const auto expert = experts_.findExpert(layerId, expertId);
            if (!expert || expert->location != MemoryTier::Vram) {
                throw std::logic_error("ready scheduler result has no resident expert");
            }
            const auto cached = payloadOffsets_.find(key(layerId, expertId));
            if (cached == payloadOffsets_.end()) {
                throw std::logic_error(
                    "resident expert has no verified storage payload offset");
            }
            payloadOffsets.push_back(cached->second);
        }
    }

    const auto& firstBinding =
        weightMap_.require(layerId, decision.selectedExpertIds.front());
    const auto& downShape = firstBinding.downProjection->shape.dimensions();
    if (downShape.size() != 2) {
        throw std::invalid_argument("expert down projection must be rank two");
    }
    const tensor::Shape outputShape{hiddenState.shape().dimensions()[0], downShape[1]};
    auto accumulated = tensorBackend_->allocateTensor(outputShape, tensor::DType::FP32);
    fillAndCopy(*tensorBackend_, accumulated, 0.0F);

    for (std::size_t index = 0; index < decision.selectedExpertIds.size(); ++index) {
        const auto expertId = decision.selectedExpertIds[index];
        const auto expert = experts_.findExpert(layerId, expertId);
        if (!expert || expert->sizeBytes == 0) {
            throw std::logic_error("selected expert metadata disappeared");
        }
        scheduler_->acquire(layerId, expertId);
        try {
            const auto payload = experts_.residentDeviceTensorView(
                layerId, expertId, tensor::Shape{expert->sizeBytes}, tensor::DType::INT8);
            const auto weights = weightMap_.createViews(
                layerId, expertId, payload, payloadOffsets[index]);
            auto expertOutput =
                tensorBackend_->allocateTensor(outputShape, tensor::DType::FP32);
            executor_->execute(hiddenState, weights, expertOutput);

            auto scale = tensorBackend_->allocateTensor(outputShape, tensor::DType::FP32);
            fillAndCopy(*tensorBackend_, scale, decision.routingScores[index]);
            auto weighted = tensorBackend_->allocateTensor(outputShape, tensor::DType::FP32);
            tensorBackend_->mul(expertOutput, scale, weighted);
            auto next = tensorBackend_->allocateTensor(outputShape, tensor::DType::FP32);
            tensorBackend_->add(accumulated, weighted, next);
            accumulated = std::move(next);
            scheduler_->release(layerId, expertId);
        } catch (...) {
            scheduler_->release(layerId, expertId);
            throw;
        }
    }
    tensorBackend_->synchronize();
    return {std::move(decision), std::move(accumulated)};
}

} // namespace hypermoe::runtime
