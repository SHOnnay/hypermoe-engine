#pragma once

#include <cstddef>
#include <span>

namespace hypermoe::transformer::position {

class RoPE {
public:
    explicit RoPE(float theta = 10000.0F);

    void apply(std::span<float> values,
               std::size_t tokenCount,
               std::size_t headCount,
               std::size_t headDimension,
               std::size_t positionOffset) const;

    [[nodiscard]] float theta() const noexcept;

private:
    float theta_{};
};

} // namespace hypermoe::transformer::position
