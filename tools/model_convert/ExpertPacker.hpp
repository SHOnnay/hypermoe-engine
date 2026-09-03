#pragma once

#include "models/ModelManifest.hpp"

#include <cstdint>
#include <filesystem>

namespace hypermoe::conversion {

struct PackingReport {
    std::uint64_t experts{};
    std::uint64_t projections{};
    std::uint64_t bytesRead{};
    std::uint64_t bytesWritten{};
};

class ExpertPacker {
public:
    [[nodiscard]] PackingReport pack(
        const models::ModelManifest& sourceManifest,
        const std::filesystem::path& artifactRoot,
        const std::filesystem::path& outputDirectory) const;
};

} // namespace hypermoe::conversion
