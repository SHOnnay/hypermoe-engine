#pragma once

#include "experts/ExpertExecutor.hpp"
#include "models/metadata/TensorMetadata.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace hypermoe::models {

enum class ExpertWeightType {
    GATE,
    UP,
    DOWN,
};

struct ExpertWeightBinding {
    std::uint32_t layerId{};
    std::uint32_t expertId{};
    std::optional<TensorMetadata> gateProjection;
    std::optional<TensorMetadata> upProjection;
    std::optional<TensorMetadata> downProjection;

    [[nodiscard]] bool complete() const noexcept;
};

class ExpertWeightMap {
public:
    void add(std::uint32_t layerId,
             std::uint32_t expertId,
             ExpertWeightType type,
             TensorMetadata tensor);
    [[nodiscard]] const ExpertWeightBinding*
    find(std::uint32_t layerId, std::uint32_t expertId) const noexcept;
    [[nodiscard]] const ExpertWeightBinding&
    require(std::uint32_t layerId, std::uint32_t expertId) const;
    [[nodiscard]] std::vector<ExpertWeightBinding> entries() const;

    [[nodiscard]] ExpertMlpWeights createViews(
        std::uint32_t layerId,
        std::uint32_t expertId,
        tensor::TensorView expertPayload,
        std::uint64_t payloadFileOffset = 0) const;

private:
    [[nodiscard]] static constexpr std::uint64_t key(std::uint32_t layerId,
                                                      std::uint32_t expertId) noexcept {
        return (static_cast<std::uint64_t>(layerId) << 32U) | expertId;
    }

    std::unordered_map<std::uint64_t, ExpertWeightBinding> mappings_;
};

} // namespace hypermoe::models
