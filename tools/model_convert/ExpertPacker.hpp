#pragma once

#include "models/ModelManifest.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

namespace hypermoe::conversion {

enum class ExpertStorageEncoding {
    Preserve,
    Int8,
};

struct ExpertPackingOptions {
    ExpertStorageEncoding expertEncoding{ExpertStorageEncoding::Preserve};
};

struct PackingReport {
    std::uint64_t layers{};
    std::uint64_t experts{};
    std::uint64_t projections{};
    std::uint64_t sourceTensors{};
    std::uint64_t parametersIndexed{};
    std::uint64_t bytesRead{};
    std::uint64_t bytesWritten{};
    std::uint64_t sourceExpertBytes{};
    std::uint64_t packedExpertBytes{};
    std::uint64_t quantizedProjections{};
    std::uint64_t shardCount{};
    float maximumQuantizationError{};
    std::string expertStorage{"PRESERVE"};
    bool validationPassed{};
    std::map<std::string, std::uint64_t> dtypeTensors;

    [[nodiscard]] std::string toJson() const;
};

class ExpertPacker {
public:
    [[nodiscard]] PackingReport pack(
        const models::ModelManifest& sourceManifest,
        const std::filesystem::path& artifactRoot,
        const std::filesystem::path& outputDirectory,
        ExpertPackingOptions options = {}) const;
};

} // namespace hypermoe::conversion
