#pragma once

#include "runtime/InferenceContext.hpp"
#include "runtime/generation/ForwardState.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace hypermoe::models::runtime {
class ModelRuntime;
}
namespace hypermoe::runtime::cache {
class KVCache;
}

namespace hypermoe::runtime::generation {

class GenerationModel {
public:
    virtual ~GenerationModel() = default;
    [[nodiscard]] virtual std::size_t vocabularySize() const noexcept = 0;
    [[nodiscard]] virtual std::size_t hiddenDimension() const noexcept = 0;
    [[nodiscard]] virtual std::size_t layerCount() const noexcept = 0;
    [[nodiscard]] virtual std::size_t keyValueHeads() const noexcept = 0;
    [[nodiscard]] virtual std::size_t headDimension() const noexcept = 0;
    [[nodiscard]] virtual ForwardPass forward(
        InferenceContext& context,
        std::span<const std::uint32_t> tokenIds,
        cache::KVCache& kvCache) = 0;
};

class ModelRuntimeGenerationModel final : public GenerationModel {
public:
    explicit ModelRuntimeGenerationModel(
        std::shared_ptr<models::runtime::ModelRuntime> runtime);

    [[nodiscard]] std::size_t vocabularySize() const noexcept override;
    [[nodiscard]] std::size_t hiddenDimension() const noexcept override;
    [[nodiscard]] std::size_t layerCount() const noexcept override;
    [[nodiscard]] std::size_t keyValueHeads() const noexcept override;
    [[nodiscard]] std::size_t headDimension() const noexcept override;
    [[nodiscard]] ForwardPass forward(
        InferenceContext& context,
        std::span<const std::uint32_t> tokenIds,
        cache::KVCache& kvCache) override;

private:
    std::shared_ptr<models::runtime::ModelRuntime> runtime_;
};

} // namespace hypermoe::runtime::generation
