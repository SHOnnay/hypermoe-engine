#pragma once

#include "models/ModelManifest.hpp"

#include <filesystem>
#include <string_view>

namespace hypermoe::importer {

class ModelImporter {
public:
    virtual ~ModelImporter() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool
    canImport(const std::filesystem::path& artifact) const noexcept = 0;
    [[nodiscard]] virtual models::ModelManifest
    inspect(const std::filesystem::path& artifact) const = 0;

    models::ModelManifest importModel(
        const std::filesystem::path& artifact,
        const std::filesystem::path& manifestPath) const {
        auto manifest = inspect(artifact);
        manifest.validate();
        manifest.save(manifestPath);
        return manifest;
    }
};

} // namespace hypermoe::importer
