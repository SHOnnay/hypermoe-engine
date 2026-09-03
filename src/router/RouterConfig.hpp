#pragma once

#include <cstddef>
#include <string_view>

namespace hypermoe::router {

enum class RoutingNormalization {
    None,
    Softmax,
};

[[nodiscard]] constexpr std::string_view
toString(RoutingNormalization normalization) noexcept {
    switch (normalization) {
    case RoutingNormalization::None: return "NONE";
    case RoutingNormalization::Softmax: return "SOFTMAX";
    }
    return "UNKNOWN";
}

struct RouterConfig {
    std::size_t expertCount{};
    std::size_t topK{1};
    RoutingNormalization normalization{RoutingNormalization::Softmax};
    bool renormalizeSelected{};

    void validate() const;
};

} // namespace hypermoe::router
