#pragma once

#include "runtime/cache/KVCache.hpp"
#include "tensor/Tensor.hpp"

#include <memory>

namespace hypermoe::tensor { class TensorBackend; }

namespace hypermoe::runtime::cache {

class CudaKVCache final : public KVCacheBase {
public:
    CudaKVCache(std::shared_ptr<tensor::TensorBackend> backend,
                std::size_t layerCount, std::size_t maximumSequenceLength,
                std::size_t keyValueHeads, std::size_t headDimension);
    void append(std::size_t layer, std::uint64_t firstPosition,
                tensor::TensorView keys, tensor::TensorView values) override;
    [[nodiscard]] KVCacheSnapshot snapshot(std::size_t layer) const override;
    [[nodiscard]] std::size_t tokenCount(std::size_t layer) const override;
    [[nodiscard]] std::size_t memoryUsageBytes() const override;
    [[nodiscard]] std::size_t maximumMemoryUsageBytes() const override;
    void clear(std::size_t layer) override;
    void reset() override;
    [[nodiscard]] std::size_t layerCount() const noexcept override;
    [[nodiscard]] std::size_t maximumSequenceLength() const noexcept override;
    [[nodiscard]] std::size_t keyValueHeads() const noexcept override;
    [[nodiscard]] std::size_t headDimension() const noexcept override;
    [[nodiscard]] tensor::Device device() const noexcept override;
private:
    struct Chunk { std::uint64_t firstPosition{}; tensor::Tensor keys; tensor::Tensor values; };
    std::shared_ptr<tensor::TensorBackend> backend_;
    std::size_t maximumSequenceLength_{};
    std::size_t keyValueHeads_{};
    std::size_t headDimension_{};
    mutable std::mutex mutex_;
    std::vector<std::vector<Chunk>> layers_;
};

} // namespace hypermoe::runtime::cache
