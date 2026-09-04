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

class KVCache {
public:
    KVCache(std::size_t layerCount,
            std::size_t maximumSequenceLength,
            std::size_t keyValueHeads,
            std::size_t headDimension);

    void append(std::size_t layer,
                std::uint64_t firstPosition,
                tensor::TensorView keys,
                tensor::TensorView values);
    [[nodiscard]] KVCacheSnapshot snapshot(std::size_t layer) const;
    [[nodiscard]] std::size_t tokenCount(std::size_t layer) const;
    [[nodiscard]] std::size_t memoryUsageBytes() const;
    void clear(std::size_t layer);
    void reset();

    [[nodiscard]] std::size_t layerCount() const noexcept;
    [[nodiscard]] std::size_t maximumSequenceLength() const noexcept;
    [[nodiscard]] std::size_t keyValueHeads() const noexcept;
    [[nodiscard]] std::size_t headDimension() const noexcept;

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
