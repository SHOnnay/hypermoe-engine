#pragma once

#include "models/ModelManifest.hpp"

#include <filesystem>
#include <vector>

namespace hypermoe::importer {

class SafeTensors {
public:
    [[nodiscard]] static std::vector<models::ManifestTensor>
    inspectFile(const std::filesystem::path& file,
                const std::filesystem::path& artifactRoot);
    [[nodiscard]] static std::vector<models::ManifestTensor>
    inspectArtifact(const std::filesystem::path& artifact);
};

} // namespace hypermoe::importer
