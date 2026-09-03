#include "models/ExpertWeightMap.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::models {

bool ExpertWeightBinding::complete() const noexcept {
    return gateProjection.has_value() && upProjection.has_value() &&
           downProjection.has_value();
}

void ExpertWeightMap::add(std::uint32_t layerId,
                          std::uint32_t expertId,
                          ExpertWeightType type,
                          TensorMetadata tensor) {
    if (!tensor.expertId || tensor.layerId != layerId ||
        *tensor.expertId != expertId) {
        throw std::invalid_argument("expert weight identity does not match mapping key");
    }
    auto& binding = mappings_[key(layerId, expertId)];
    binding.layerId = layerId;
    binding.expertId = expertId;
    std::optional<TensorMetadata>* target = nullptr;
    switch (type) {
    case ExpertWeightType::GATE: target = &binding.gateProjection; break;
    case ExpertWeightType::UP: target = &binding.upProjection; break;
    case ExpertWeightType::DOWN: target = &binding.downProjection; break;
    }
    if (target == nullptr) throw std::invalid_argument("unknown expert weight type");
    if (target->has_value()) throw std::invalid_argument("duplicate expert projection");
    *target = std::move(tensor);
}

const ExpertWeightBinding*
ExpertWeightMap::find(std::uint32_t layerId, std::uint32_t expertId) const noexcept {
    const auto found = mappings_.find(key(layerId, expertId));
    return found == mappings_.end() ? nullptr : &found->second;
}

const ExpertWeightBinding&
ExpertWeightMap::require(std::uint32_t layerId, std::uint32_t expertId) const {
    const auto* binding = find(layerId, expertId);
    if (binding == nullptr) throw std::out_of_range("expert weight mapping not found");
    if (!binding->complete()) throw std::logic_error("expert weight mapping is incomplete");
    return *binding;
}

std::vector<ExpertWeightBinding> ExpertWeightMap::entries() const {
    std::vector<ExpertWeightBinding> result;
    result.reserve(mappings_.size());
    for (const auto& [mappingKey, binding] : mappings_) {
        (void)mappingKey;
        result.push_back(binding);
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        if (left.layerId != right.layerId) return left.layerId < right.layerId;
        return left.expertId < right.expertId;
    });
    return result;
}

ExpertMlpWeights ExpertWeightMap::createViews(
    std::uint32_t layerId,
    std::uint32_t expertId,
    tensor::TensorView expertPayload,
    std::uint64_t payloadFileOffset) const {
    const auto& binding = require(layerId, expertId);
    const auto makeView = [&](const TensorMetadata& metadata) {
        if (metadata.isQuantized()) {
            throw std::runtime_error("quantized expert execution is not implemented");
        }
        if (metadata.offset < payloadFileOffset) {
            throw std::invalid_argument("tensor offset precedes expert payload");
        }
        const auto relative = metadata.offset - payloadFileOffset;
        if (relative > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("tensor offset exceeds addressable memory");
        }
        const auto elementBytes = tensor::sizeOf(metadata.dtype);
        if (elementBytes == 0 ||
            metadata.shape.storageElementCount() >
                std::numeric_limits<std::uint64_t>::max() / elementBytes ||
            metadata.shape.storageElementCount() * elementBytes != metadata.size) {
            throw std::invalid_argument("tensor metadata size does not match shape and dtype");
        }
        return expertPayload.sliceBytes(static_cast<std::size_t>(relative),
                                        metadata.shape, metadata.dtype);
    };
    return {makeView(*binding.gateProjection), makeView(*binding.upProjection),
            makeView(*binding.downProjection)};
}

} // namespace hypermoe::models
