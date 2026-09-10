#include "runtime/cache/CudaKVCache.hpp"

#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"

#include <cstring>
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
    std::size_t tokens{};
    for (const auto& chunk : layers_[layer]) tokens += chunk.keys.shape().dimensions()[0];
    if (first != tokens || keys.shape().dimensions()[0] > maximumSequenceLength_ - tokens) {
        throw std::invalid_argument("CUDA KV cache append exceeds contiguous capacity");
    }
    auto ownedKeys = backend_->allocateTensor(keys.shape(), tensor::DType::FP32);
    auto ownedValues = backend_->allocateTensor(values.shape(), tensor::DType::FP32);
    backend_->copyTensor(keys, ownedKeys);
    backend_->copyTensor(values, ownedValues);
    layers_[layer].push_back({first, std::move(ownedKeys), std::move(ownedValues)});
}

KVCacheSnapshot CudaKVCache::snapshot(std::size_t layer) const {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) throw std::out_of_range("CUDA KV cache layer is invalid");
    KVCacheSnapshot result;
    result.keyValueHeads = keyValueHeads_;
    result.headDimension = headDimension_;
    tensor::CpuTensorBackend cpu;
    for (const auto& chunk : layers_[layer]) {
        auto hostKeys = cpu.allocateTensor(chunk.keys.shape(), tensor::DType::FP32);
        auto hostValues = cpu.allocateTensor(chunk.values.shape(), tensor::DType::FP32);
        backend_->copyTensor(chunk.keys.view(), hostKeys.view());
        backend_->copyTensor(chunk.values.view(), hostValues.view());
        const auto count = chunk.keys.shape().dimensions()[0];
        for (std::size_t token = 0; token < count; ++token) {
            result.positions.push_back(chunk.firstPosition + token);
        }
        const auto* keys = static_cast<const float*>(hostKeys.data());
        const auto* values = static_cast<const float*>(hostValues.data());
        result.keys.insert(result.keys.end(), keys, keys + hostKeys.shape().elementCount());
        result.values.insert(result.values.end(), values, values + hostValues.shape().elementCount());
    }
    return result;
}

std::size_t CudaKVCache::tokenCount(std::size_t layer) const {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) {
        throw std::out_of_range("CUDA KV cache layer is invalid");
    }
    std::size_t tokens{};
    for (const auto& chunk : layers_[layer]) {
        tokens += chunk.keys.shape().dimensions()[0];
    }
    return tokens;
}
std::size_t CudaKVCache::memoryUsageBytes() const {
    std::scoped_lock lock(mutex_);
    std::size_t bytes{};
    for (const auto& layer : layers_) {
        for (const auto& chunk : layer) {
            const auto chunkBytes = chunk.keys.bytes() + chunk.values.bytes() +
                chunk.keys.shape().dimensions()[0] * sizeof(std::uint64_t);
            if (bytes > std::numeric_limits<std::size_t>::max() - chunkBytes) {
                throw std::overflow_error("CUDA KV cache byte accounting overflows");
            }
            bytes += chunkBytes;
        }
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
    layers_[layer].clear();
}
void CudaKVCache::reset() {
    std::scoped_lock lock(mutex_);
    for (auto& layer : layers_) layer.clear();
}
std::size_t CudaKVCache::layerCount() const noexcept { return layers_.size(); }
std::size_t CudaKVCache::maximumSequenceLength() const noexcept { return maximumSequenceLength_; }
std::size_t CudaKVCache::keyValueHeads() const noexcept { return keyValueHeads_; }
std::size_t CudaKVCache::headDimension() const noexcept { return headDimension_; }
tensor::Device CudaKVCache::device() const noexcept { return backend_->device(); }

} // namespace hypermoe::runtime::cache
