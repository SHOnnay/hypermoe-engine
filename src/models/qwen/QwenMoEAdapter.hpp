#pragma once

#include "models/ModelAdapter.hpp"

namespace hypermoe::models::qwen {

class QwenMoEAdapter final : public ModelAdapter {
public:
    [[nodiscard]] ModelMetadata
    loadModelMetadata(const std::filesystem::path& manifestPath) const override;
    [[nodiscard]] std::vector<TensorMetadata>
    loadTensorIndex(const std::filesystem::path& manifestPath) const override;
    [[nodiscard]] ModelArchitecture getArchitecture() const noexcept override;
    [[nodiscard]] ModelCapabilities capabilities() const noexcept override;
    [[nodiscard]] std::size_t
    getLayerCount(const ModelMetadata& metadata) const noexcept override;
    [[nodiscard]] std::size_t
    getExpertCount(const ModelMetadata& metadata) const noexcept override;
    [[nodiscard]] router::RouterConfig
    getRouterConfiguration(const ModelMetadata& metadata) const override;
    [[nodiscard]] ExpertWeightMap
    getExpertMapping(const ModelMetadata& metadata) const override;
};

} // namespace hypermoe::models::qwen
