#include "runtime/cache/KVCache.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace hypermoe::runtime::cache {

std::size_t KVCacheSnapshot::tokenCount() const noexcept {
    return positions.size();
}

KVCache::KVCache(std::size_t layerCount,
                 std::size_t maximumSequenceLength,
                 std::size_t keyValueHeads,
                 std::size_t headDimension)
    : maximumSequenceLength_(maximumSequenceLength),
      keyValueHeads_(keyValueHeads),
      headDimension_(headDimension),
      layers_(layerCount) {
    if (layerCount == 0 || maximumSequenceLength_ == 0 || keyValueHeads_ == 0 ||
        headDimension_ == 0 ||
        keyValueHeads_ > std::numeric_limits<std::size_t>::max() / headDimension_) {
        throw std::invalid_argument("KV cache dimensions must be nonzero");
    }
}

void KVCache::append(std::size_t layer,
                     std::uint64_t firstPosition,
                     tensor::TensorView keys,
                     tensor::TensorView values) {
    [[maybe_unused]] const auto keyOwner = keys.lockOwner();
    [[maybe_unused]] const auto valueOwner = values.lockOwner();
    if (!keyOwner || !valueOwner || layer >= layers_.size() || !keys || !values ||
        keys.device() != tensor::Device::cpu() ||
        values.device() != tensor::Device::cpu() ||
        keys.dtype() != tensor::DType::FP32 ||
        values.dtype() != tensor::DType::FP32 || !keys.isContiguous() ||
        !values.isContiguous() || keys.shape() != values.shape() ||
        keys.shape().rank() != 3) {
        throw std::invalid_argument("KV cache append requires matching CPU FP32 tensors");
    }
    const auto& dimensions = keys.shape().dimensions();
    if (dimensions[0] == 0 || dimensions[1] != keyValueHeads_ ||
        dimensions[2] != headDimension_) {
        throw std::invalid_argument("KV cache tensor shape is incompatible");
    }
    std::scoped_lock lock(mutex_);
    auto& storage = layers_[layer];
    if (firstPosition != storage.positions.size() ||
        dimensions[0] > maximumSequenceLength_ - storage.positions.size()) {
        throw std::invalid_argument(
            "KV cache append must be contiguous and within sequence capacity");
    }
    const auto oldSize = storage.keys.size();
    const auto appended = keys.shape().elementCount();
    storage.keys.resize(oldSize + appended);
    storage.values.resize(oldSize + appended);
    std::memcpy(storage.keys.data() + oldSize, keys.data(), keys.bytes());
    std::memcpy(storage.values.data() + oldSize, values.data(), values.bytes());
    for (std::size_t token = 0; token < dimensions[0]; ++token) {
        storage.positions.push_back(firstPosition + token);
    }
}

KVCacheSnapshot KVCache::snapshot(std::size_t layer) const {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) throw std::out_of_range("KV cache layer is invalid");
    const auto& storage = layers_[layer];
    return {storage.positions, storage.keys, storage.values,
            keyValueHeads_, headDimension_};
}

std::size_t KVCache::tokenCount(std::size_t layer) const {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) throw std::out_of_range("KV cache layer is invalid");
    return layers_[layer].positions.size();
}

std::size_t KVCache::memoryUsageBytes() const {
    std::scoped_lock lock(mutex_);
    std::size_t result{};
    for (const auto& layer : layers_) {
        result += layer.positions.size() * sizeof(std::uint64_t);
        result += (layer.keys.size() + layer.values.size()) * sizeof(float);
    }
    return result;
}

void KVCache::clear(std::size_t layer) {
    std::scoped_lock lock(mutex_);
    if (layer >= layers_.size()) throw std::out_of_range("KV cache layer is invalid");
    layers_[layer] = {};
}

void KVCache::reset() {
    std::scoped_lock lock(mutex_);
    for (auto& layer : layers_) layer = {};
}

std::size_t KVCache::layerCount() const noexcept { return layers_.size(); }
std::size_t KVCache::maximumSequenceLength() const noexcept {
    return maximumSequenceLength_;
}
std::size_t KVCache::keyValueHeads() const noexcept { return keyValueHeads_; }
std::size_t KVCache::headDimension() const noexcept { return headDimension_; }

} // namespace hypermoe::runtime::cache
