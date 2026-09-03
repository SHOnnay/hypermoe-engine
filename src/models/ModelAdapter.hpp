#pragma once

#include "models/ExpertWeightMap.hpp"
#include "models/ModelMetadata.hpp"

#include <filesystem>
#include <vector>

namespace hypermoe::models {

class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;

    [[nodiscard]] virtual ModelMetadata
    loadModelMetadata(const std::filesystem::path& manifestPath) const = 0;
    [[nodiscard]] virtual std::vector<TensorMetadata>
    loadTensorIndex(const std::filesystem::path& manifestPath) const = 0;
    [[nodiscard]] virtual ModelArchitecture getArchitecture() const noexcept = 0;
    [[nodiscard]] virtual ModelCapabilities capabilities() const noexcept = 0;
    [[nodiscard]] virtual std::size_t
    getLayerCount(const ModelMetadata& metadata) const noexcept = 0;
    [[nodiscard]] virtual std::size_t
    getExpertCount(const ModelMetadata& metadata) const noexcept = 0;
    [[nodiscard]] virtual router::RouterConfig
    getRouterConfiguration(const ModelMetadata& metadata) const = 0;
    [[nodiscard]] virtual ExpertWeightMap
    getExpertMapping(const ModelMetadata& metadata) const = 0;
};

} // namespace hypermoe::models
