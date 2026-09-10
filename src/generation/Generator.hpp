#pragma once

#include "generation/LogitsProcessor.hpp"
#include "runtime/generation/GenerationState.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hypermoe::runtime::cache {
class KVCacheManager;
}
namespace hypermoe::runtime::generation {
class GenerationModel;
}
namespace hypermoe::tokenizer {
class Tokenizer;
}

namespace hypermoe::generation {

struct GenerationConfig {
    std::size_t maximumNewTokens{1};
    SamplingConfig sampling;
    std::vector<std::uint32_t> stopTokenIds;
};

struct GenerationMetrics {
    std::chrono::nanoseconds tokenization{};
    std::chrono::nanoseconds prefill{};
    std::chrono::nanoseconds decode{};
    std::chrono::nanoseconds total{};
    std::size_t decodeSteps{};
    std::size_t peakKVCacheBytes{};

    [[nodiscard]] double decodeTokensPerSecond() const noexcept;
};

struct GenerationResult {
    std::string text;
    std::vector<std::uint32_t> promptTokens;
    std::vector<std::uint32_t> generatedTokens;
    runtime::generation::GenerationStopReason stopReason{
        runtime::generation::GenerationStopReason::None};
    GenerationMetrics metrics;
};

class Generator {
public:
    Generator(std::shared_ptr<tokenizer::Tokenizer> tokenizer,
              std::shared_ptr<runtime::generation::GenerationModel> model,
              std::shared_ptr<runtime::cache::KVCacheManager> cacheManager);

    [[nodiscard]] GenerationResult generate(
        std::string_view prompt,
        const GenerationConfig& config) const;

private:
    std::shared_ptr<tokenizer::Tokenizer> tokenizer_;
    std::shared_ptr<runtime::generation::GenerationModel> model_;
    std::shared_ptr<runtime::cache::KVCacheManager> cacheManager_;
};

} // namespace hypermoe::generation
