#include "transformer/attention/CudaAttention.hpp"

#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"
#include "runtime/cache/CudaKVCache.hpp"
#include "transformer/attention/CpuAttention.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::transformer::attention {
namespace {

tensor::Tensor toHost(tensor::TensorBackend& backend, tensor::TensorView value) {
    [[maybe_unused]] const auto owner = value.lockOwner();
    if (!owner || !value || value.device() != backend.device() ||
        !value.isContiguous()) {
        throw std::invalid_argument("CUDA attention received an incompatible tensor");
    }
    tensor::CpuTensorBackend cpu;
    auto result = cpu.allocateTensor(value.shape(), value.dtype());
    backend.copyTensor(value, result.view());
    return result;
}

tensor::Tensor toDevice(tensor::TensorBackend& backend, tensor::TensorView value) {
    auto result = backend.allocateTensor(value.shape(), value.dtype());
    backend.copyTensor(value, result.view());
    return result;
}

} // namespace

CudaAttention::CudaAttention(std::shared_ptr<tensor::TensorBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available() ||
        backend_->device().type != tensor::DeviceType::CUDA) {
        throw std::invalid_argument("CUDA attention requires an available CUDA backend");
    }
}

std::string_view CudaAttention::name() const noexcept {
    return "CUDA projected reference attention";
}
tensor::Device CudaAttention::device() const noexcept { return backend_->device(); }

AttentionResult CudaAttention::execute(
    tensor::TensorView hiddenStates,
    const AttentionWeights& weights,
    const AttentionConfiguration& configuration) {
    auto* cuda = dynamic_cast<tensor::CudaTensorBackend*>(backend_.get());
    if (cuda && cuda->nativeKernelsAvailable()) {
        [[maybe_unused]] const auto hiddenOwner = hiddenStates.lockOwner();
        [[maybe_unused]] const auto queryOwner = weights.query.lockOwner();
        [[maybe_unused]] const auto keyOwner = weights.key.lockOwner();
        [[maybe_unused]] const auto valueOwner = weights.value.lockOwner();
        [[maybe_unused]] const auto outputOwner = weights.output.lockOwner();
        [[maybe_unused]] const auto queryNormOwner = weights.queryNorm.lockOwner();
        [[maybe_unused]] const auto keyNormOwner = weights.keyNorm.lockOwner();
        const auto matrix = [&](tensor::TensorView value) {
            return value && value.device() == device() &&
                   value.dtype() == tensor::DType::FP32 && value.isContiguous() &&
                   value.shape().rank() == 2;
        };
        if (!hiddenOwner || !queryOwner || !keyOwner || !valueOwner || !outputOwner ||
            !matrix(hiddenStates) || !matrix(weights.query) || !matrix(weights.key) ||
            !matrix(weights.value) || !matrix(weights.output)) {
            throw std::invalid_argument("CUDA attention requires device FP32 matrices");
        }
        auto headDimension = configuration.headDimension;
                const auto projectionHeadDimension = configuration.projectionHeadDimension;
                const auto& hiddenShape = hiddenStates.shape().dimensions();
                const auto& queryShape = weights.query.shape().dimensions();
                const auto& keyShape = weights.key.shape().dimensions();
                const auto& valueShape = weights.value.shape().dimensions();
                const auto& outputShape = weights.output.shape().dimensions();
                if (headDimension == 0) {
                    if (configuration.headCount == 0 ||
                        queryShape[1] % configuration.headCount != 0) {
                        throw std::invalid_argument("CUDA attention cannot infer head dimension");
                    }
                    headDimension = queryShape[1] / configuration.headCount;
                }
                if (configuration.headCount == 0 || configuration.keyValueHeadCount == 0 ||
                    headDimension == 0 || configuration.headCount % configuration.keyValueHeadCount != 0 ||
                    configuration.headCount > std::numeric_limits<std::size_t>::max() / headDimension ||
                    configuration.keyValueHeadCount > std::numeric_limits<std::size_t>::max() / headDimension) {
                    throw std::invalid_argument("CUDA attention head configuration is invalid");
                }
                const auto queryWidth = configuration.headCount * projectionHeadDimension;
                const auto keyValueWidth =
                    configuration.keyValueHeadCount * projectionHeadDimension;
                if (queryShape != std::vector<std::size_t>{hiddenShape[1], queryWidth} ||
                    keyShape != std::vector<std::size_t>{hiddenShape[1], keyValueWidth} ||
                    valueShape != keyShape ||
                    outputShape != std::vector<std::size_t>{queryWidth, hiddenShape[1]} ||
                    !std::isfinite(configuration.ropeTheta) || configuration.ropeTheta <= 0.0F ||
                    configuration.positionOffset > std::numeric_limits<std::size_t>::max()) {
                    throw std::invalid_argument("CUDA attention projections are incompatible");
                }
        const auto tokenCount = hiddenShape[0];
        AttentionResult result;
        result.query = backend_->allocateTensor({tokenCount, queryWidth}, tensor::DType::FP32);
        result.key = backend_->allocateTensor({tokenCount, keyValueWidth}, tensor::DType::FP32);
        result.value = backend_->allocateTensor({tokenCount, keyValueWidth}, tensor::DType::FP32);
        backend_->matmul(hiddenStates, weights.query, result.query.view());
        backend_->matmul(hiddenStates, weights.key, result.key.view());
        backend_->matmul(hiddenStates, weights.value, result.value.view());
        if (static_cast<bool>(weights.queryNorm) !=
            static_cast<bool>(weights.keyNorm)) {
            throw std::invalid_argument(
                "CUDA attention requires both query and key normalization weights");
        }
        if (weights.queryNorm) {
                    const auto normWeight = [&](tensor::TensorView value,
                                                const std::shared_ptr<void>& owner) {
                        return owner && value.device() == device() &&
                               value.dtype() == tensor::DType::FP32 &&
                               value.isContiguous() &&
                               value.shape() == tensor::Shape{configuration.projectionHeadDimension};
                    };
            if (!normWeight(weights.queryNorm, queryNormOwner) ||
                !normWeight(weights.keyNorm, keyNormOwner) ||
                !std::isfinite(configuration.queryKeyNormEpsilon) ||
                configuration.queryKeyNormEpsilon <= 0.0F) {
                throw std::invalid_argument(
                    "CUDA attention Q/K normalization is incompatible");
            }
            auto normalizedQuery = backend_->allocateTensor(
                            result.query.shape(), tensor::DType::FP32);
                        auto normalizedKey = backend_->allocateTensor(
                            result.key.shape(), tensor::DType::FP32);
                        cuda->rmsNorm(
                            result.query.view().reshape(
                                {tokenCount * configuration.headCount, configuration.projectionHeadDimension}),
                            weights.queryNorm,
                            normalizedQuery.view().reshape(
                                {tokenCount * configuration.headCount, configuration.projectionHeadDimension}),
                            configuration.queryKeyNormEpsilon);
                        cuda->rmsNorm(
                            result.key.view().reshape(
                                {tokenCount * configuration.keyValueHeadCount, configuration.projectionHeadDimension}),
                            weights.keyNorm,
                            normalizedKey.view().reshape(
                                {tokenCount * configuration.keyValueHeadCount, configuration.projectionHeadDimension}),
                            configuration.queryKeyNormEpsilon);
                        result.query = std::move(normalizedQuery);
                        result.key = std::move(normalizedKey);
                    }
                    if (configuration.rotaryEmbedding) {
                        cuda->applyRoPE(result.query.view(), tokenCount, configuration.headCount,
                                        headDimension,
                                        static_cast<std::size_t>(configuration.positionOffset),
                                        configuration.ropeTheta);
                        cuda->applyRoPE(result.key.view(), tokenCount,
                                        configuration.keyValueHeadCount, headDimension,
                                        static_cast<std::size_t>(configuration.positionOffset),
                                        configuration.ropeTheta);
        }

        hypermoe::runtime::cache::CudaKVDeviceSnapshot cached;
        tensor::TensorView keys = result.key.view().reshape(
            {tokenCount, configuration.keyValueHeadCount, headDimension});
        tensor::TensorView values = result.value.view().reshape(
            {tokenCount, configuration.keyValueHeadCount, headDimension});
        auto keyPositionOffset = configuration.positionOffset;
        if (configuration.kvCache) {
            auto* cache = dynamic_cast<hypermoe::runtime::cache::CudaKVCache*>(
                configuration.kvCache);
            if (!cache || cache->device() != device() ||
                cache->keyValueHeads() != configuration.keyValueHeadCount ||
                cache->headDimension() != headDimension) {
                throw std::invalid_argument(
                    "native CUDA attention requires a compatible CUDA KV cache");
            }
            cache->append(configuration.layerIndex, configuration.positionOffset,
                          keys, values);
            cached = cache->deviceSnapshot(configuration.layerIndex);
            keys = cached.keys;
            values = cached.values;
            keyPositionOffset = cached.firstPosition;
        }
        const auto keyTokens = keys.shape().dimensions()[0];
        result.scores = backend_->allocateTensor(
            {configuration.headCount, tokenCount, keyTokens}, tensor::DType::FP32);
        result.probabilities = backend_->allocateTensor(
            result.scores.shape(), tensor::DType::FP32);
        result.context = backend_->allocateTensor(
            {tokenCount, queryWidth}, tensor::DType::FP32);
        result.output = backend_->allocateTensor(
            {tokenCount, hiddenShape[1]}, tensor::DType::FP32);
        cuda->causalAttention(
            result.query.view(), keys, values, result.scores.view(),
            result.probabilities.view(), result.context.view(),
            configuration.headCount, configuration.keyValueHeadCount,
            headDimension, configuration.positionOffset, keyPositionOffset,
            configuration.causal);
        backend_->matmul(result.context.view(), weights.output, result.output.view());
        return result;
    }

    auto hostBackend = std::make_shared<tensor::CpuTensorBackend>();
    auto hostHidden = toHost(*backend_, hiddenStates);
    auto hostQueryWeights = toHost(*backend_, weights.query);
    auto hostKeyWeights = toHost(*backend_, weights.key);
    auto hostValueWeights = toHost(*backend_, weights.value);
    auto hostOutputWeights = toHost(*backend_, weights.output);
    tensor::Tensor hostQueryNorm;
    tensor::Tensor hostKeyNorm;
    if (weights.queryNorm) {
        if (!weights.keyNorm) {
            throw std::invalid_argument(
                "CUDA attention requires both query and key normalization weights");
        }
        hostQueryNorm = toHost(*backend_, weights.queryNorm);
        hostKeyNorm = toHost(*backend_, weights.keyNorm);
    } else if (weights.keyNorm) {
        throw std::invalid_argument(
            "CUDA attention requires both query and key normalization weights");
    }
    CpuAttention reference(hostBackend);
    auto host = reference.execute(
        hostHidden.view(),
        {hostQueryWeights.view(), hostKeyWeights.view(), hostValueWeights.view(),
         hostOutputWeights.view(), hostQueryNorm.view(), hostKeyNorm.view()},
        configuration);

    AttentionResult result;
    result.query = backend_->allocateTensor(host.query.shape(), host.query.dtype());
    result.key = backend_->allocateTensor(host.key.shape(), host.key.dtype());
    result.value = backend_->allocateTensor(host.value.shape(), host.value.dtype());
    result.output = backend_->allocateTensor(host.output.shape(), host.output.dtype());

    // Toolkit-only fallback: projections execute through cuBLAS while the
    // reference path supplies RoPE, masking, softmax, and context.
    backend_->matmul(hiddenStates, weights.query, result.query.view());
    backend_->matmul(hiddenStates, weights.key, result.key.view());
    backend_->matmul(hiddenStates, weights.value, result.value.view());
    if (configuration.rotaryEmbedding || weights.queryNorm) {
        backend_->copyTensor(host.query.view(), result.query.view());
        backend_->copyTensor(host.key.view(), result.key.view());
    }
    result.scores = toDevice(*backend_, host.scores.view());
    result.probabilities = toDevice(*backend_, host.probabilities.view());
    result.context = toDevice(*backend_, host.context.view());
    backend_->matmul(result.context.view(), weights.output, result.output.view());
    backend_->synchronize();
    return result;
}

} // namespace hypermoe::transformer::attention
