#pragma once

#include "models/metadata/TensorMetadata.hpp"

#include <cstdint>
#include <vector>

namespace hypermoe::models {

struct ExpertMetadata {
    std::uint32_t layerId{};
    std::uint32_t expertId{};
    std::vector<TensorMetadata> tensors;
};

} // namespace hypermoe::models
