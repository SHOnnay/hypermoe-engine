#pragma once

#include <cstddef>

namespace hypermoe::models {
class ModelManifest;
}

namespace hypermoe::models::runtime {

enum class NormalizationKind {
    RMSNorm,
};

struct NormalizationConfiguration {
    NormalizationKind kind{NormalizationKind::RMSNorm};
    float epsilon{1.0e-6F};

    void validate() const;
};

struct ModelArchitecture {
    std::size_t layerCount{};
    std::size_t hiddenDimension{};
    std::size_t attentionHeads{1};
    std::size_t keyValueHeads{1};
    std::size_t headDimension{};
    std::size_t expertCount{};
    std::size_t topK{1};
    float ropeTheta{10000.0F};
    NormalizationConfiguration inputNormalization;
    NormalizationConfiguration postAttentionNormalization;

    void validate() const;
    [[nodiscard]] static ModelArchitecture fromManifest(
        const models::ModelManifest& manifest);
};

} // namespace hypermoe::models::runtime
