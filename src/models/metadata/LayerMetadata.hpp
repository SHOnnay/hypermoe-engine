#pragma once

#include "models/metadata/ExpertMetadata.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace hypermoe::models {

struct LayerMetadata {
    std::uint32_t layerId{};
    std::vector<ExpertMetadata> experts;
    std::optional<TensorMetadata> routerTensor;
};

} // namespace hypermoe::models
