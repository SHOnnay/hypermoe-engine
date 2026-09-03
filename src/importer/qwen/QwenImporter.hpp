#pragma once

#include "importer/ModelImporter.hpp"

namespace hypermoe::importer::qwen {

class QwenImporter final : public ModelImporter {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool
    canImport(const std::filesystem::path& artifact) const noexcept override;
    [[nodiscard]] models::ModelManifest
    inspect(const std::filesystem::path& artifact) const override;
};

} // namespace hypermoe::importer::qwen
