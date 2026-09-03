#include "router/RouterConfig.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hypermoe::router {

void RouterConfig::validate() const {
    if (expertCount == 0) throw std::invalid_argument("router requires experts");
    if (expertCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("router expert count exceeds runtime IDs");
    }
    if (topK == 0 || topK > expertCount) {
        throw std::invalid_argument("router top-k must be within expert count");
    }
    switch (normalization) {
    case RoutingNormalization::None:
    case RoutingNormalization::Softmax: return;
    }
    throw std::invalid_argument("router normalization is unsupported");
}

} // namespace hypermoe::router
