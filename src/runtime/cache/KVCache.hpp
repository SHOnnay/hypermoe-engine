#pragma once

#include "tensor/TensorView.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace hypermoe::runtime::cache {

struct KVCacheSnapshot {
    std::vector<std::uint64_t> positions;
    std::vector<float> keys;
    std::vector<float> values;
    std::size_t keyValueHeads{};
    std::size_t headDimension{};

    [[nodiscard]] std::size_t tokenCount() const noexcept;
};

class KVCacheBase {
public:
    virtual ~KVCacheBase() = default;
    virtual void append(std::size_t layer, std::uint64_t firstPosition,
                        tensor::TensorView keys, tensor::TensorView values) = 0;
    [[nodiscard]] virtual KVCacheSnapshot snapshot(std::size_t layer) const = 0;
    [[nodiscard]] virtual std::size_t tokenCount(std::size_t layer) const = 0;
    [[nodiscard]] virtual std::size_t memoryUsageBytes() const = 0;
    [[nodiscard]] virtual std::size_t maximumMemoryUsageBytes() const = 0;
    virtual void clear(std::size_t layer) = 0;
    virtual void reset() = 0;
    [[nodiscard]] virtual std::size_t layerCount() const noexcept = 0;
    [[nodiscard]] virtual std::size_t maximumSequenceLength() const noexcept = 0;
    [[nodiscard]] virtual std::size_t keyValueHeads() const noexcept = 0;
    [[nodiscard]] virtual std::size_t headDimension() const noexcept = 0;
    [[nodiscard]] virtual tensor::Device device() const noexcept = 0;
};

class KVCache final : public KVCacheBase {
public:
    KVCache(std::size_t layerCount,
            std::size_t maximumSequenceLength,
            std::size_t keyValueHeads,
            std::size_t headDimension);

    void append(std::size_t layer,
                std::uint64_t firstPosition,
                tensor::TensorView keys,
                tensor::TensorView values) override;
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
    struct LayerStorage {
        std::vector<std::uint64_t> positions;
        std::vector<float> keys;
        std::vector<float> values;
    };

    std::size_t maximumSequenceLength_{};
    std::size_t keyValueHeads_{};
    std::size_t headDimension_{};
    mutable std::mutex mutex_;
    std::vector<LayerStorage> layers_;
};

} // namespace hypermoe::runtime::cache
