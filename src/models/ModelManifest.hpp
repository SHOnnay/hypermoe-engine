#pragma once

#include "models/ModelConfig.hpp"
#include "models/runtime/ModelArchitecture.hpp"
#include "router/RouterConfig.hpp"
#include "tensor/DType.hpp"
#include "tensor/Shape.hpp"
#include "tensor/quantization/Quantization.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hypermoe::models {

enum class TensorLayout : std::uint32_t {
    InputOutput = 0,
    OutputInput = 1,
};

[[nodiscard]] constexpr bool isValid(TensorLayout layout) noexcept {
    return layout == TensorLayout::InputOutput ||
           layout == TensorLayout::OutputInput;
}

[[nodiscard]] constexpr std::string_view toString(TensorLayout layout) noexcept {
    switch (layout) {
    case TensorLayout::InputOutput: return "INPUT_OUTPUT";
    case TensorLayout::OutputInput: return "OUTPUT_INPUT";
    }
    return "UNKNOWN";
}

struct ManifestTensor {
    ManifestTensor() = default;
    ManifestTensor(
        std::string tensorName,
        std::filesystem::path tensorSourceFile,
        std::uint64_t tensorOffset,
        std::uint64_t tensorSize,
        tensor::DType tensorDType,
        tensor::Shape tensorShape,
        std::optional<tensor::quantization::QuantizationParameters>
            tensorQuantization = {})
        : name(std::move(tensorName)),
          sourceFile(std::move(tensorSourceFile)),
          offset(tensorOffset),
          size(tensorSize),
          dtype(tensorDType),
          shape(std::move(tensorShape)),
          quantization(tensorQuantization) {}

    std::string name;
    std::filesystem::path sourceFile;
    std::uint64_t offset{};
    std::uint64_t size{};
    tensor::DType dtype{tensor::DType::FP32};
    tensor::Shape shape;
    std::optional<tensor::quantization::QuantizationParameters> quantization;
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
    std::string queryNormTensor{};
    std::string keyNormTensor{};
};

struct ManifestModelIO {
    std::string tokenEmbeddingTensor;
    std::string finalNormTensor;
    ManifestTensorBinding lmHead;
    bool tiedEmbeddings{};
};

class ModelManifest {
public:
    static constexpr std::string_view schemaVersion =
        "hypermoe.model-manifest.v3";
    static constexpr std::string_view legacySchemaVersion =
        "hypermoe.model-manifest.v2";

    std::string schema{schemaVersion};
    std::string modelName;
    ModelArchitecture architecture{ModelArchitecture::UNKNOWN};
    std::string sourceArchitecture;
    std::uint64_t parameterCount{};
    ModelConfig config;
    std::optional<runtime::ModelArchitecture> runtimeArchitecture;
    ManifestRouter router;
    std::vector<ManifestTensor> tensors;
    std::vector<ManifestExpertMapping> experts;
    std::vector<ManifestLayerMapping> layers;
    std::optional<ManifestModelIO> modelIO;

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
