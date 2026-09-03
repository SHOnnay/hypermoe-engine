#pragma once

#include "models/ModelConfig.hpp"
#include "models/metadata/LayerMetadata.hpp"
#include "router/RouterConfig.hpp"

#include <string>
#include <vector>

namespace hypermoe::models {

struct ModelMetadata {
    std::string schema{"hypermoe.model-manifest.v1"};
    std::string architectureName;
    ModelConfig config;
    router::RouterConfig router;
    std::vector<TensorMetadata> tensors;
    std::vector<LayerMetadata> layers;
};

} // namespace hypermoe::models
