#include "models/runtime/ModelArchitecture.hpp"

#include "models/ModelManifest.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hypermoe::models::runtime {

void NormalizationConfiguration::validate() const {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument("normalization epsilon must be positive and finite");
    }
}

void ModelArchitecture::validate() const {
    if (layerCount == 0 || hiddenDimension == 0 || attentionHeads == 0 ||
        keyValueHeads == 0 || headDimension == 0 || headDimension % 2 != 0 ||
        expertCount == 0 || topK == 0 ||
        topK > expertCount || attentionHeads % keyValueHeads != 0 ||
        attentionHeads > std::numeric_limits<std::size_t>::max() / headDimension ||
        attentionHeads * headDimension != hiddenDimension ||
        !std::isfinite(ropeTheta) || ropeTheta <= 0.0F) {
        throw std::invalid_argument("model architecture dimensions are invalid");
    }
    inputNormalization.validate();
    postAttentionNormalization.validate();
}

ModelArchitecture ModelArchitecture::fromManifest(
    const models::ModelManifest& manifest) {
    manifest.validate();
    ModelArchitecture result;
    if (manifest.runtimeArchitecture) {
        result = *manifest.runtimeArchitecture;
    } else {
        result.layerCount = manifest.config.layerCount;
        result.hiddenDimension = manifest.config.hiddenSize;
        result.attentionHeads = 1;
        result.keyValueHeads = 1;
        result.headDimension = manifest.config.hiddenSize;
        result.expertCount = manifest.config.expertCount;
        result.topK = manifest.router.config.topK;
    }
    result.validate();
    if (result.layerCount != manifest.config.layerCount ||
        result.hiddenDimension != manifest.config.hiddenSize ||
        result.expertCount != manifest.config.expertCount ||
        result.topK != manifest.router.config.topK) {
        throw std::invalid_argument(
            "runtime architecture disagrees with manifest model dimensions");
    }
    return result;
}

} // namespace hypermoe::models::runtime
