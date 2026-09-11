#pragma once

#include "importer/validation/CheckpointValidator.hpp"
#include "models/ModelManifest.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace hypermoe::importer::qwen {

struct QwenTokenizerMetadata {
    std::string tokenizerClass;
    std::string tokenizerModel;
    std::size_t declaredVocabularySize{};
    std::size_t vocabularyEntries{};
    std::size_t addedTokens{};
    std::vector<std::uint32_t> bosTokenIds;
    std::vector<std::uint32_t> eosTokenIds;
    std::vector<std::uint32_t> padTokenIds;
    bool hasChatTemplate{};
    bool hasSpecialTokensMap{};

    void validate() const;
    [[nodiscard]] std::string toJson() const;
};

struct QwenCheckpointDescriptor {
    std::filesystem::path root;
    models::ModelManifest manifest;
    validation::CheckpointValidationReport validation;
    QwenTokenizerMetadata tokenizer;
};

class QwenCheckpointLoader {
public:
    [[nodiscard]] QwenCheckpointDescriptor load(
        const std::filesystem::path& artifact) const;
};

} // namespace hypermoe::importer::qwen
