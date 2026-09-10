#include "runtime/cache/CudaKVCache.hpp"

#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::runtime::cache {

CudaKVCache::CudaKVCache(std::shared_ptr<tensor::TensorBackend> backend,
                         std::size_t layers, std::size_t maximumSequence,
                         std::size_t heads, std::size_t dimension)
    : backend_(std::move(backend)), maximumSequenceLength_(maximumSequence),
      keyValueHeads_(heads), headDimension_(dimension), layers_(layers) {
    if (!backend_ || !backend_->available() ||
        backend_->device().type != tensor::DeviceType::CUDA) {
        throw std::invalid_argument("CUDA KV cache requires an available CUDA backend");
    }
    KVCache sizing(layers, maximumSequence, heads, dimension);
    (void)sizing.maximumMemoryUsageBytes();
}

void CudaKVCache::append(std::size_t layer, std::uint64_t first,
                         tensor::TensorView keys, tensor::TensorView values) {
    if (layer >= layers_.size() || !keys || !values || keys.shape() != values.shape() ||
        keys.dtype() != tensor::DType::FP32 || values.dtype() != tensor::DType::FP32 ||
        !keys.isContiguous() || !values.isContiguous() || keys.shape().rank() != 3 ||
        keys.shape().dimensions()[1] != keyValueHeads_ ||
        keys.shape().dimensions()[2] != headDimension_) {
        throw std::invalid_argument("CUDA KV cache append metadata is incompatible");
    }
    std::scoped_lock lock(mutex_);
    auto& storage = layers_[layer];
    const auto incoming = keys.shape().dimensions()[0];
    if (first != storage.tokens || incoming > maximumSequenceLength_ - storage.tokens) {
        throw std::invalid_argument("CUDA KV cache append exceeds contiguous capacity");
    }
    const auto required = storage.tokens + incoming;
    if (storage.capacity < required) {
        auto capacity = std::max<std::size_t>(1, storage.capacity);
        while (capacity < required) {
            capacity = capacity > maximumSequenceLength_ / 2U
                ? maximumSequenceLength_ : capacity * 2U;
        }
        auto expandedKeys = backend_->allocateTensor(
            {capacity, keyValueHeads_, headDimension_}, tensor::DType::FP32);
        auto expandedValues = backend_->allocateTensor(
            expandedKeys.shape(), tensor::DType::FP32);
        if (storage.tokens != 0) {
            const tensor::Shape logical{
                storage.tokens, keyValueHeads_, headDimension_};
            backend_->copyTensor(
                storage.keys.view().sliceBytes(0, logical, tensor::DType::FP32),
                expandedKeys.view().sliceBytes(0, logical, tensor::DType::FP32));
            backend_->copyTensor(
                storage.values.view().sliceBytes(0, logical, tensor::DType::FP32),
                expandedValues.view().sliceBytes(0, logical, tensor::DType::FP32));
        }
        storage.keys = std::move(expandedKeys);
        storage.values = std::move(expandedValues);
        storage.capacity = capacity;
    }
    const auto offset = storage.tokens * keyValueHeads_ * headDimension_ * sizeof(float);
    backend_->copyTensor(
        keys, storage.keys.view().sliceBytes(offset, keys.shape(), tensor::DType::FP32));
    backend_->copyTensor(
        values, storage.values.view().sliceBytes(offset, values.shape(), tensor::DType::FP32));
    if (storage.tokens == 0) storage.firstPosition = first;
    storage.tokens = required;
}

KVCacheSnapshot CudaKVCache::snapshot(std::size_t layer) const {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) throw std::out_of_range("CUDA KV cache layer is invalid");
    KVCacheSnapshot result;
    result.keyValueHeads = keyValueHeads_;
    result.headDimension = headDimension_;
    const auto& storage = layers_[layer];
    if (storage.tokens == 0) return result;
    const tensor::Shape logical{storage.tokens, keyValueHeads_, headDimension_};
    tensor::CpuTensorBackend cpu;
    auto hostKeys = cpu.allocateTensor(logical, tensor::DType::FP32);
    auto hostValues = cpu.allocateTensor(logical, tensor::DType::FP32);
    backend_->copyTensor(
        storage.keys.view().sliceBytes(0, logical, tensor::DType::FP32),
        hostKeys.view());
    backend_->copyTensor(
        storage.values.view().sliceBytes(0, logical, tensor::DType::FP32),
        hostValues.view());
    for (std::size_t token = 0; token < storage.tokens; ++token) {
        result.positions.push_back(storage.firstPosition + token);
    }
    const auto* keys = static_cast<const float*>(hostKeys.data());
    const auto* values = static_cast<const float*>(hostValues.data());
    result.keys.assign(keys, keys + hostKeys.shape().elementCount());
    result.values.assign(values, values + hostValues.shape().elementCount());
    return result;
}

CudaKVDeviceSnapshot CudaKVCache::deviceSnapshot(std::size_t layer) const {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) {
        throw std::out_of_range("CUDA KV cache layer is invalid");
    }
    const auto& storage = layers_[layer];
    if (storage.tokens == 0) {
        throw std::logic_error("CUDA KV cache layer has no device snapshot");
    }
    const tensor::Shape logical{storage.tokens, keyValueHeads_, headDimension_};
    CudaKVDeviceSnapshot result;
    result.keys = storage.keys.view().sliceBytes(0, logical, tensor::DType::FP32);
    result.values = storage.values.view().sliceBytes(0, logical, tensor::DType::FP32);
    result.tokenCount = storage.tokens;
    result.firstPosition = storage.firstPosition;
    return result;
}

std::size_t CudaKVCache::tokenCount(std::size_t layer) const {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) {
        throw std::out_of_range("CUDA KV cache layer is invalid");
    }
    return layers_[layer].tokens;
}
std::size_t CudaKVCache::memoryUsageBytes() const {
    std::scoped_lock lock(mutex_);
    std::size_t bytes{};
    for (const auto& layer : layers_) {
        const auto layerBytes = (layer.keys ? layer.keys.bytes() : 0) +
            (layer.values ? layer.values.bytes() : 0) +
            layer.tokens * sizeof(std::uint64_t);
        if (bytes > std::numeric_limits<std::size_t>::max() - layerBytes) {
            throw std::overflow_error("CUDA KV cache byte accounting overflows");
        }
        bytes += layerBytes;
    }
    return bytes;
}
std::size_t CudaKVCache::maximumMemoryUsageBytes() const {
    KVCache sizing(layers_.size(), maximumSequenceLength_, keyValueHeads_, headDimension_);
    return sizing.maximumMemoryUsageBytes();
}
void CudaKVCache::clear(std::size_t layer) {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) {
        throw std::out_of_range("CUDA KV cache layer is invalid");
    }
    layers_[layer] = {};
}
void CudaKVCache::reset() {
    std::scoped_lock lock(mutex_);
    for (auto& layer : layers_) layer = {};
}
std::size_t CudaKVCache::layerCount() const noexcept { return layers_.size(); }
std::size_t CudaKVCache::maximumSequenceLength() const noexcept { return maximumSequenceLength_; }
std::size_t CudaKVCache::keyValueHeads() const noexcept { return keyValueHeads_; }
std::size_t CudaKVCache::headDimension() const noexcept { return headDimension_; }
tensor::Device CudaKVCache::device() const noexcept { return backend_->device(); }

} // namespace hypermoe::runtime::cache
