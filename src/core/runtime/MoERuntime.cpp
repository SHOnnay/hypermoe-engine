#include "core/runtime/MoERuntime.hpp"

#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"
#include "tensor/precision/DTypeConverter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hypermoe::runtime {

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

LayerExecutionResult MoERuntime::executeLayer(LayerId layerId,
                                              tensor::TensorView hiddenState,
                                              tensor::TensorView routerWeights) {
    if (!hiddenState || hiddenState.shape().rank() != 2 ||
        hiddenState.shape().dimensions()[0] != 1) {
        throw std::invalid_argument("single-layer execution requires exactly one token");
    }
    auto batch = executeBatch(layerId, hiddenState, routerWeights);
    auto routing = std::move(batch.routing.tokens.front());
    return {std::move(routing), std::move(batch.output)};
}

BatchLayerExecutionResult MoERuntime::executeBatch(
    LayerId layerId,
    tensor::TensorView hiddenStates,
    tensor::TensorView routerWeights) {
    std::scoped_lock executionLock(executionMutex_);
    if (!hiddenStates || hiddenStates.shape().rank() != 2 ||
        hiddenStates.dtype() != tensor::DType::FP32 ||
        hiddenStates.device() != tensorBackend_->device()) {
        throw std::invalid_argument(
            "MoE runtime requires rank-2 FP32 hidden states on its tensor backend");
    }
    ExecutionMetadata metadata;
    metadata.tensorBackend = std::string(tensorBackend_->name());
    const auto routingStart = std::chrono::steady_clock::now();
    auto decision = router_->routeBatch(layerId, hiddenStates, routerWeights);
    metadata.routingTime = std::chrono::steady_clock::now() - routingStart;
    if (!decision.valid() ||
        decision.tokens.size() != hiddenStates.shape().dimensions()[0]) {
        throw std::runtime_error("router returned an invalid batch decision");
    }
    for (const auto& tokenDecision : decision.tokens) {
        if (predictor_ && history_) {
            (void)predictor_->observeAndPrefetch(
                tokenDecision, *history_, *scheduler_);
        } else {
            if (history_) history_->record(tokenDecision);
            if (predictor_) predictor_->observe(tokenDecision);
        }
    }

    std::vector<ExpertBatch> batches;
    batches.reserve(decision.expertGroups.size());
    for (const auto& group : decision.expertGroups) {
        ExpertBatch batch{layerId, group.expertId, group.tokenIndices,
                          group.routingScores};
        batch.validate(decision.tokens.size());
        metadata.expertAssignments += batch.size();
        batches.push_back(std::move(batch));
    }
    if (batches.empty()) throw std::runtime_error("router produced no expert batches");
    std::size_t expectedAssignments{};
    for (const auto& token : decision.tokens) {
        if (expectedAssignments >
            std::numeric_limits<std::size_t>::max() -
                token.selectedExpertIds.size()) {
            throw std::overflow_error("router assignment count overflow");
        }
        expectedAssignments += token.selectedExpertIds.size();
    }
    if (metadata.expertAssignments != expectedAssignments) {
        throw std::runtime_error("router expert groups omit token assignments");
    }
    metadata.uniqueExperts = batches.size();

    const auto schedulingStart = std::chrono::steady_clock::now();
    std::vector<scheduler::ScheduleHandle> handles;
    handles.reserve(batches.size());
    for (const auto& batch : batches) {
        const auto expert = experts_.findExpert(layerId, batch.expertId);
        if (!expert) throw std::out_of_range("router selected an unregistered expert");
        if (metadata.expertPayloadBytes >
            std::numeric_limits<std::uint64_t>::max() - expert->sizeBytes) {
            throw std::overflow_error("expert payload byte count overflow");
        }
        metadata.expertPayloadBytes += expert->sizeBytes;
        scheduler::ScheduleRequest request;
        request.layerId = layerId;
        request.expertId = batch.expertId;
        request.source = expert->location;
        request.destination = MemoryTier::Vram;
        request.priority = scheduler::TransferPriority::ActiveInference;
        handles.push_back(scheduler_->schedule(std::move(request)));
    }

    std::vector<std::uint64_t> payloadOffsets;
    payloadOffsets.reserve(handles.size());
    for (std::size_t index = 0; index < handles.size(); ++index) {
        const auto& scheduled = handles[index].future().get();
        const auto expertId = batches[index].expertId;
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
    metadata.schedulingTime = std::chrono::steady_clock::now() - schedulingStart;

    const auto& firstBinding =
        weightMap_.require(layerId, batches.front().expertId);
    const auto& downShape = firstBinding.downProjection->shape.dimensions();
    if (downShape.size() != 2) {
        throw std::invalid_argument("expert down projection must be rank two");
    }
    const auto tokenCount = hiddenStates.shape().dimensions()[0];
    const auto inputWidth = hiddenStates.shape().dimensions()[1];
    const auto outputWidth = downShape[1];
    const tensor::Shape outputShape{tokenCount, outputWidth};
    tensor::CpuTensorBackend cpu;
    auto hostHidden = cpu.allocateTensor(hiddenStates.shape(), tensor::DType::FP32);
    tensorBackend_->copyTensor(hiddenStates, hostHidden);
    auto hostCombined = cpu.allocateTensor(outputShape, tensor::DType::FP32);
    std::fill_n(static_cast<float*>(hostCombined.data()),
                hostCombined.shape().elementCount(), 0.0F);
    const auto* hiddenValues = static_cast<const float*>(hostHidden.data());
    auto* combinedValues = static_cast<float*>(hostCombined.data());
    std::vector<tensor::Tensor> expertOutputs;
    expertOutputs.reserve(batches.size());

    for (std::size_t index = 0; index < batches.size(); ++index) {
        const auto& batch = batches[index];
        const auto expertId = batch.expertId;
        const auto expert = experts_.findExpert(layerId, expertId);
        if (!expert || expert->sizeBytes == 0) {
            throw std::logic_error("selected expert metadata disappeared");
        }
        scheduler_->acquire(layerId, expertId);
        try {
            const auto expertStart = std::chrono::steady_clock::now();
            auto residency = experts_.acquireResidentExpert(layerId, expertId);
            const auto payload = residency.view(
                tensor::Shape{expert->sizeBytes}, tensor::DType::INT8);
            const auto weights = weightMap_.createViews(
                layerId, expertId, payload, payloadOffsets[index]);
            std::array<tensor::Tensor, 3> converted;
            ExpertMlpWeights executionWeights = weights;
            const auto prepare = [&](tensor::TensorView source,
                                     tensor::Tensor& owner) {
                if (source.dtype() == tensor::DType::FP32) return source;
                owner = tensor::precision::DTypeConverter::toFp32Tensor(
                    source, *tensorBackend_);
                return owner.view();
            };
            executionWeights.gateProjection =
                prepare(weights.gateProjection, converted[0]);
            executionWeights.upProjection =
                prepare(weights.upProjection, converted[1]);
            executionWeights.downProjection =
                prepare(weights.downProjection, converted[2]);
            auto hostExpertInput = cpu.allocateTensor(
                {batch.size(), inputWidth}, tensor::DType::FP32);
            auto* groupInput = static_cast<float*>(hostExpertInput.data());
            for (std::size_t row = 0; row < batch.size(); ++row) {
                std::memcpy(groupInput + row * inputWidth,
                            hiddenValues + batch.tokenIndices[row] * inputWidth,
                            inputWidth * sizeof(float));
            }
            auto expertInput = tensorBackend_->allocateTensor(
                hostExpertInput.shape(), tensor::DType::FP32);
            tensorBackend_->copyTensor(hostExpertInput, expertInput);
            auto expertOutput = tensorBackend_->allocateTensor(
                {batch.size(), outputWidth}, tensor::DType::FP32);
            executor_->execute(expertInput, executionWeights, expertOutput);
            metadata.expertExecutionTime +=
                std::chrono::steady_clock::now() - expertStart;

            const auto combinationStart = std::chrono::steady_clock::now();
            auto hostExpertOutput = cpu.allocateTensor(
                expertOutput.shape(), tensor::DType::FP32);
            tensorBackend_->copyTensor(expertOutput, hostExpertOutput);
            const auto* expertValues =
                static_cast<const float*>(hostExpertOutput.data());
            for (std::size_t row = 0; row < batch.size(); ++row) {
                const auto token = batch.tokenIndices[row];
                const auto weight = batch.routingWeights[row];
                for (std::size_t hidden = 0; hidden < outputWidth; ++hidden) {
                    combinedValues[token * outputWidth + hidden] +=
                        weight * expertValues[row * outputWidth + hidden];
                }
            }
            metadata.expertCombinationTime +=
                std::chrono::steady_clock::now() - combinationStart;
            expertOutputs.push_back(std::move(expertOutput));
            scheduler_->release(layerId, expertId);
        } catch (...) {
            scheduler_->release(layerId, expertId);
            throw;
        }
    }
    const auto outputCopyStart = std::chrono::steady_clock::now();
    auto output = tensorBackend_->allocateTensor(outputShape, tensor::DType::FP32);
    tensorBackend_->copyTensor(hostCombined, output);
    tensorBackend_->synchronize();
    metadata.expertCombinationTime +=
        std::chrono::steady_clock::now() - outputCopyStart;
    return {std::move(decision), std::move(batches), std::move(expertOutputs),
            std::move(output), std::move(metadata)};
}

} // namespace hypermoe::runtime
