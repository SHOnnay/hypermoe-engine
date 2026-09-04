#pragma once

#include "models/ModelConfig.hpp"
#include "models/runtime/ModelArchitecture.hpp"
#include "router/RouterConfig.hpp"
#include "tensor/DType.hpp"
#include "tensor/Shape.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace hypermoe::models {

enum class TensorLayout {
    InputOutput,
    OutputInput,
};

[[nodiscard]] constexpr std::string_view toString(TensorLayout layout) noexcept {
    return layout == TensorLayout::InputOutput ? "INPUT_OUTPUT" : "OUTPUT_INPUT";
}

struct ManifestTensor {
    std::string name;
    std::filesystem::path sourceFile;
    std::uint64_t offset{};
    std::uint64_t size{};
    tensor::DType dtype{tensor::DType::FP32};
    tensor::Shape shape;
};

struct ProjectionLocation {
    std::string tensorName;
    std::uint64_t offset{};
    std::uint64_t size{};
    tensor::Shape shape;
    TensorLayout layout{TensorLayout::OutputInput};
};

struct ManifestExpertMapping {
    std::uint32_t layerId{};
    std::uint32_t expertId{};
    ProjectionLocation gate;
    ProjectionLocation up;
    ProjectionLocation down;
};

struct ManifestRouterTensor {
    std::uint32_t layerId{};
    std::string tensorName;
};

struct ManifestRouter {
    router::RouterConfig config;
    std::vector<ManifestRouterTensor> tensors;
    TensorLayout layout{TensorLayout::OutputInput};
};

struct ManifestTensorBinding {
    std::string tensorName;
    TensorLayout layout{TensorLayout::OutputInput};
};

struct ManifestLayerMapping {
    std::uint32_t layerId{};
    ManifestTensorBinding queryProjection;
    ManifestTensorBinding keyProjection;
    ManifestTensorBinding valueProjection;
    ManifestTensorBinding outputProjection;
    std::string inputNormTensor;
    std::string postAttentionNormTensor;
    std::string routerTensor;
};

class ModelManifest {
public:
    static constexpr std::string_view schemaVersion =
        "hypermoe.model-manifest.v2";

    std::string schema{schemaVersion};
    std::string modelName;
    ModelArchitecture architecture{ModelArchitecture::UNKNOWN};
    std::string sourceArchitecture;
    ModelConfig config;
    std::optional<runtime::ModelArchitecture> runtimeArchitecture;
    ManifestRouter router;
    std::vector<ManifestTensor> tensors;
    std::vector<ManifestExpertMapping> experts;
    std::vector<ManifestLayerMapping> layers;

    void validate() const;
    [[nodiscard]] const ManifestTensor* findTensor(std::string_view name) const noexcept;
    [[nodiscard]] const ManifestExpertMapping*
    findExpert(std::uint32_t layerId, std::uint32_t expertId) const noexcept;
    [[nodiscard]] const ManifestLayerMapping*
    findLayer(std::uint32_t layerId) const noexcept;
    [[nodiscard]] std::string toJson() const;
    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static ModelManifest load(const std::filesystem::path& path);
};

} // namespace hypermoe::models
