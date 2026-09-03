#pragma once

#include "models/ModelManifest.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

namespace hypermoe::importer::validation {

struct CheckpointValidationReport {
    std::size_t shardCount{};
    std::size_t tensorCount{};
    std::size_t referencedTensorCount{};
    std::size_t expertCount{};
    std::size_t routerTensorCount{};
    std::uint64_t referencedBytes{};
    std::map<std::string, std::size_t> dtypeCounts;
};

class CheckpointValidator {
public:
    [[nodiscard]] static CheckpointValidationReport validate(
        const std::filesystem::path& artifact,
        const models::ModelManifest& manifest);
};

} // namespace hypermoe::importer::validation
