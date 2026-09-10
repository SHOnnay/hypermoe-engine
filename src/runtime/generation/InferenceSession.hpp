#pragma once

#include "runtime/cache/KVCacheManager.hpp"
#include "runtime/generation/ForwardState.hpp"
#include "runtime/generation/GenerationState.hpp"
#include "runtime/generation/InferenceConfig.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace hypermoe::runtime::generation {

class GenerationModel;

class InferenceSession {
public:
    InferenceSession(std::shared_ptr<GenerationModel> model,
                     std::shared_ptr<cache::KVCacheManager> cacheManager,
                     std::size_t maximumNewTokens,
                     std::span<const std::uint32_t> stopTokenIds = {},
                     InferenceConfig config = {});
    ~InferenceSession();

    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;
    InferenceSession(InferenceSession&&) = delete;
    InferenceSession& operator=(InferenceSession&&) = delete;

    [[nodiscard]] GenerationModel& model() const noexcept;
    [[nodiscard]] cache::KVCacheBase& kvCache() const noexcept;
    [[nodiscard]] cache::KVCacheSessionId id() const noexcept;
    [[nodiscard]] GenerationState& generationState() noexcept;
    [[nodiscard]] const GenerationState& generationState() const noexcept;
    [[nodiscard]] ForwardState& forwardState() noexcept;
    [[nodiscard]] const ForwardState& forwardState() const noexcept;

private:
    std::shared_ptr<GenerationModel> model_;
    std::shared_ptr<cache::KVCacheManager> cacheManager_;
    cache::KVCacheSessionId id_{};
    std::shared_ptr<cache::KVCacheBase> kvCache_;
    GenerationState generationState_;
    ForwardState forwardState_;
};

} // namespace hypermoe::runtime::generation
