#include "core/runtime/MoERuntime.hpp"

#include "experts/ExpertExecutor.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
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
    std::shared_ptr<prediction::ExpertPredictor> predictor,
    bool transferComputeOverlap)
    : router_(std::move(router)),
      scheduler_(std::move(scheduler)),
      experts_(experts),
      weightMap_(std::move(weightMap)),
      tensorBackend_(std::move(tensorBackend)),
      executor_(std::move(executor)),
      history_(std::move(history)),
      predictor_(std::move(predictor)),
      transferComputeOverlap_(transferComputeOverlap) {
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
    scheduler_->expirePrefetchesBefore(layerId);
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
    for (std::size_t tokenIndex = 0; tokenIndex < decision.tokens.size(); ++tokenIndex) {
        const auto& tokenDecision = decision.tokens[tokenIndex];
        if (predictor_ && history_) {
            (void)predictor_->observeAndPrefetch(
                tokenDecision, *history_, *scheduler_, tokenIndex,
                [this](const prediction::ExpertPrediction& prediction) {
                    try {
                        experts_.updatePrediction(prediction.expectedLayer,
                                                  prediction.expertId,
                                                  prediction.probability,
                                                  prediction.confidence);
                    } catch (const std::out_of_range&) {
                        // Partial runtime graphs may omit predicted experts.
                    }
                });
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
    auto output = tensorBackend_->allocateTensor(outputShape, tensor::DType::FP32);
    auto* cuda = dynamic_cast<tensor::CudaTensorBackend*>(tensorBackend_.get());
    const auto nativeCuda = cuda && cuda->nativeKernelsAvailable();
    const auto deviceExecution = tensorBackend_->device().type == tensor::DeviceType::CUDA;
    const auto destination = deviceExecution ? MemoryTier::Vram : MemoryTier::Ram;
    std::vector<scheduler::ScheduleHandle> handles(batches.size());
    const auto submit = [&](std::size_t index) {
        const auto schedulingStart = std::chrono::steady_clock::now();
        const auto expertId = batches[index].expertId;
        auto expert = experts_.findExpert(layerId, expertId);
        if (!expert) throw std::out_of_range("router selected an unregistered expert");
        const auto predicted = scheduler_->state(layerId, expertId);
        if (expert->location == MemoryTier::Nvme &&
            predicted.targetLocation == MemoryTier::Ram &&
            (predicted.state == scheduler::ExpertLifecycleState::Queued ||
             predicted.state == scheduler::ExpertLifecycleState::Loading ||
             predicted.state == scheduler::ExpertLifecycleState::Ready)) {
            scheduler::ScheduleRequest warmRequest;
            warmRequest.layerId = layerId;
            warmRequest.expertId = expertId;
            warmRequest.destination = MemoryTier::Ram;
            const auto warmHandle = scheduler_->schedule(std::move(warmRequest));
            const auto& warm = warmHandle.future().get();
            if (!warm.success) {
                throw std::runtime_error("warm expert scheduling failed: " + warm.error);
            }
        }
        // Consume warm prediction results through the existing ownership path.
        if (const auto warm = scheduler_->cachedTransfer(layerId, expertId);
            warm && warm->buffer && expert->location == MemoryTier::Nvme) {
            experts_.adoptHostWeights(layerId, expertId, warm->buffer);
            payloadOffsets_[key(layerId, expertId)] = warm->record.offset;
            scheduler_->acknowledgeResidency(layerId, expertId, MemoryTier::Ram);
            expert = experts_.findExpert(layerId, expertId);
        }
        scheduler_->reconcileResidency(layerId, expertId, expert->location);
        experts_.prepareResidency(layerId, expertId, destination);
        scheduler::ScheduleRequest request;
        request.layerId = layerId;
        request.expertId = expertId;
        request.source = expert->location;
        request.destination = destination;
        request.hostBuffer = experts_.residentWeights(layerId, expertId);
        request.deviceBuffer = experts_.residentDeviceWeights(layerId, expertId);
        handles[index] = scheduler_->schedule(std::move(request));
        metadata.schedulingTime += std::chrono::steady_clock::now() - schedulingStart;
    };
    submit(0);
    tensor::CpuTensorBackend cpu;
    tensor::Tensor hostHidden;
    tensor::Tensor hostCombined;
    const float* hiddenValues{};
    float* combinedValues{};
    if (nativeCuda) {
        cuda->zero(output.view());
    } else {
        hostHidden = cpu.allocateTensor(hiddenStates.shape(), tensor::DType::FP32);
        tensorBackend_->copyTensor(hiddenStates, hostHidden.view());
        hostCombined = cpu.allocateTensor(outputShape, tensor::DType::FP32);
        std::fill_n(static_cast<float*>(hostCombined.data()),
                    hostCombined.shape().elementCount(), 0.0F);
        hiddenValues = static_cast<const float*>(hostHidden.data());
        combinedValues = static_cast<float*>(hostCombined.data());
    }
    std::vector<tensor::Tensor> expertOutputs;
    expertOutputs.reserve(batches.size());

    for (std::size_t index = 0; index < batches.size(); ++index) {
        const auto& batch = batches[index];
        const auto expertId = batch.expertId;
        if (!handles[index].valid()) submit(index);
        const auto schedulingStart = std::chrono::steady_clock::now();
        // Wait ONLY for the expert being consumed, not the complete routed set.
        const auto& scheduled = handles[index].future().get();
        if (!scheduled.success) {
            throw std::runtime_error("expert scheduling failed: " + scheduled.error);
        }
        const auto expertBefore = experts_.findExpert(layerId, expertId);
        if (!expertBefore) throw std::logic_error("expert metadata disappeared");
        if (expertBefore->location == destination) ++metadata.expertCacheHits;
        else ++metadata.expertCacheMisses;
        metadata.expertTransferBytes += deviceExecution
            ? scheduled.transfer.ramToVramBytes : scheduled.transfer.nvmeBytes;
        metadata.expertPayloadBytes += expertBefore->sizeBytes;
        if (deviceExecution && scheduled.transfer.deviceBuffer) {
            experts_.adoptDeviceWeights(layerId, expertId, scheduled.transfer.deviceBuffer);
            payloadOffsets_[key(layerId, expertId)] = scheduled.transfer.record.offset;
        } else if (!deviceExecution && scheduled.transfer.buffer) {
            experts_.adoptHostWeights(layerId, expertId, scheduled.transfer.buffer);
            payloadOffsets_[key(layerId, expertId)] = scheduled.transfer.record.offset;
        }
        scheduler_->acknowledgeResidency(layerId, expertId, destination);
        const auto offset = payloadOffsets_.find(key(layerId, expertId));
        if (offset == payloadOffsets_.end()) {
            throw std::logic_error("resident expert has no verified storage payload offset");
        }
        const auto payloadOffset = offset->second;
        handles[index] = {}; // Scheduler no longer retains evicted physical buffers.
        metadata.schedulingTime += std::chrono::steady_clock::now() - schedulingStart;
        const auto expert = experts_.findExpert(layerId, expertId);
        if (!expert || expert->sizeBytes == 0) {
            throw std::logic_error("selected expert metadata disappeared");
        }
        scheduler_->acquire(layerId, expertId);
        ExpertResidencyLease residency;
        try {
            const auto expertStart = std::chrono::steady_clock::now();
            residency = deviceExecution
                ? experts_.acquireResidentExpert(layerId, expertId)
                : experts_.acquireHostExpert(layerId, expertId);
            const auto payload = residency.view(
                tensor::Shape{expert->sizeBytes}, tensor::DType::INT8);
            if (transferComputeOverlap_ && index + 1 < batches.size()) {
                const auto next = experts_.findExpert(layerId, batches[index + 1].expertId);
                const auto nextState = scheduler_->state(layerId, batches[index + 1].expertId);
                const auto unfinishedWarmLoad = next && next->location == MemoryTier::Nvme &&
                    nextState.targetLocation == MemoryTier::Ram &&
                    !scheduler_->cachedTransfer(layerId, next->id).has_value();
                const auto memory = experts_.memorySnapshot();
                const auto limit = deviceExecution ? memory.vram.limitBytes : memory.ram.limitBytes;
                // Never require two experts to fit when the configured budget
                // can hold only one. The current lease protects its allocation.
                if (next && !unfinishedWarmLoad && expert->sizeBytes <= limit &&
                    next->sizeBytes <= limit - expert->sizeBytes) {
                    submit(index + 1);
                }
            }
            const auto weights = weightMap_.createViews(
                layerId, expertId, payload, payloadOffset);
            std::array<tensor::Tensor, 3> converted;
            ExpertMlpWeights executionWeights = weights;
            const auto prepare = [&](tensor::TensorView source,
                                     tensor::Tensor& owner) {
                if (source.dtype() == tensor::DType::FP32 ||
                    source.dtype() == tensor::DType::INT8) {
                    return source;
                }
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
            tensor::Tensor expertInput;
            if (nativeCuda) {
                expertInput = tensorBackend_->allocateTensor(
                    {batch.size(), inputWidth}, tensor::DType::FP32);
                cuda->gatherRows(hiddenStates, batch.tokenIndices,
                                 expertInput.view());
            } else {
                auto hostExpertInput = cpu.allocateTensor(
                    {batch.size(), inputWidth}, tensor::DType::FP32);
                auto* groupInput = static_cast<float*>(hostExpertInput.data());
                for (std::size_t row = 0; row < batch.size(); ++row) {
                    std::memcpy(groupInput + row * inputWidth,
                                hiddenValues + batch.tokenIndices[row] * inputWidth,
                                inputWidth * sizeof(float));
                }
                expertInput = tensorBackend_->allocateTensor(
                    hostExpertInput.shape(), tensor::DType::FP32);
                tensorBackend_->copyTensor(hostExpertInput.view(), expertInput.view());
            }
            auto expertOutput = tensorBackend_->allocateTensor(
                {batch.size(), outputWidth}, tensor::DType::FP32);
            executor_->execute(expertInput.view(), executionWeights,
                               expertOutput.view());
            metadata.expertExecutionTime +=
                std::chrono::steady_clock::now() - expertStart;

            const auto combinationStart = std::chrono::steady_clock::now();
            if (nativeCuda) {
                cuda->scatterAddRows(expertOutput.view(), batch.tokenIndices,
                                     batch.routingWeights, output.view());
            } else {
                auto hostExpertOutput = cpu.allocateTensor(
                    expertOutput.shape(), tensor::DType::FP32);
                tensorBackend_->copyTensor(expertOutput.view(), hostExpertOutput.view());
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
            }
            metadata.expertCombinationTime +=
                std::chrono::steady_clock::now() - combinationStart;
            // scatterAddRows (native CUDA) or output readback (fallback)
            // completes the compute dependency before returning. Keep the lease
            // through that boundary; do not add a second compute-stream wait.
            expertOutputs.push_back(std::move(expertOutput));
            scheduler_->release(layerId, expertId);
        } catch (...) {
            // Keep the lease alive through error-path completion as well.
            if (deviceExecution) tensorBackend_->synchronizeExecution();
            scheduler_->release(layerId, expertId);
            throw;
        }
    }
    const auto outputCopyStart = std::chrono::steady_clock::now();
    if (!nativeCuda) {
        tensorBackend_->copyTensor(hostCombined.view(), output.view());
    }
    if (!nativeCuda) tensorBackend_->synchronizeExecution();
    metadata.expertCombinationTime +=
        std::chrono::steady_clock::now() - outputCopyStart;
    return {std::move(decision), std::move(batches), std::move(expertOutputs),
            std::move(output), std::move(metadata)};
}

} // namespace hypermoe::runtime
