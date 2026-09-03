#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace hypermoe::models {

enum class ModelArchitecture {
    QWEN_MOE,
    DEEPSEEK_MOE,
    GLM_MOE,
    KIMI_MOE,
    MIXTRAL_MOE,
    UNKNOWN,
};

[[nodiscard]] constexpr std::string_view
toString(ModelArchitecture architecture) noexcept {
    switch (architecture) {
    case ModelArchitecture::QWEN_MOE: return "QWEN_MOE";
    case ModelArchitecture::DEEPSEEK_MOE: return "DEEPSEEK_MOE";
    case ModelArchitecture::GLM_MOE: return "GLM_MOE";
    case ModelArchitecture::KIMI_MOE: return "KIMI_MOE";
    case ModelArchitecture::MIXTRAL_MOE: return "MIXTRAL_MOE";
    case ModelArchitecture::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

struct ModelCapabilities {
    bool routedExperts{};
    bool configurableTopK{};
    bool normalizedRouting{};
    bool gatedExpertMlp{};
    bool sharedExperts{};
    bool quantizedExpertWeights{};
};

struct ModelConfig {
    std::string modelName;
    std::size_t layerCount{};
    std::size_t expertCount{};
    std::size_t hiddenSize{};
    std::size_t intermediateSize{};
    ModelCapabilities capabilities;
};

} // namespace hypermoe::models
