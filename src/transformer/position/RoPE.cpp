#include "transformer/position/RoPE.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hypermoe::transformer::position {

RoPE::RoPE(float theta) : theta_(theta) {
    if (!std::isfinite(theta_) || theta_ <= 0.0F) {
        throw std::invalid_argument("RoPE theta must be positive and finite");
    }
}

void RoPE::apply(std::span<float> values,
                 std::size_t tokenCount,
                 std::size_t headCount,
                 std::size_t headDimension,
                 std::size_t positionOffset) const {
    if (tokenCount == 0 || headCount == 0 || headDimension == 0 ||
        headDimension % 2 != 0 ||
        headCount > std::numeric_limits<std::size_t>::max() / headDimension ||
        tokenCount > std::numeric_limits<std::size_t>::max() /
                         (headCount * headDimension) ||
        values.size() != tokenCount * headCount * headDimension ||
        positionOffset > std::numeric_limits<std::size_t>::max() - tokenCount) {
        throw std::invalid_argument("RoPE tensor dimensions are invalid");
    }
    for (std::size_t token = 0; token < tokenCount; ++token) {
        const auto position = static_cast<double>(positionOffset + token);
        for (std::size_t head = 0; head < headCount; ++head) {
            const auto base = (token * headCount + head) * headDimension;
            for (std::size_t dimension = 0; dimension < headDimension;
                 dimension += 2) {
                const auto exponent = static_cast<double>(dimension) /
                                      static_cast<double>(headDimension);
                const auto angle = position / std::pow(theta_, exponent);
                const auto cosine = static_cast<float>(std::cos(angle));
                const auto sine = static_cast<float>(std::sin(angle));
                const auto first = values[base + dimension];
                const auto second = values[base + dimension + 1];
                values[base + dimension] = first * cosine - second * sine;
                values[base + dimension + 1] = first * sine + second * cosine;
            }
        }
    }
}

float RoPE::theta() const noexcept { return theta_; }

} // namespace hypermoe::transformer::position
