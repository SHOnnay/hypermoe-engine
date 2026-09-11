#include "importer/qwen/QwenCheckpointLoader.hpp"

#include "importer/qwen/QwenImporter.hpp"
#include "models/metadata/JsonValue.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace hypermoe::importer::qwen {
namespace {

using models::metadata::JsonValue;
using models::metadata::MetadataError;

std::string escape(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const auto character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        if (character == '\n') {
            result += "\\n";
        } else if (character == '\r') {
            result += "\\r";
        } else if (character == '\t') {
            result += "\\t";
        } else {
            result.push_back(character);
        }
    }
    return result;
}

std::filesystem::path rootFor(const std::filesystem::path& artifact) {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(artifact, error);
    if (error) throw MetadataError("Qwen checkpoint does not exist");
    const auto directory = std::filesystem::is_directory(canonical, error);
    if (error) throw MetadataError("cannot inspect Qwen checkpoint path");
    return directory ? canonical : canonical.parent_path();
}

std::vector<std::uint32_t> tokenIds(const JsonValue& config,
                                    std::string_view name,
                                    std::size_t vocabularySize) {
    const auto* value = config.find(name);
    if (!value || value->isNull()) return {};
    std::vector<std::uint32_t> result;
    const auto append = [&](const JsonValue& token) {
        const auto id = token.asUInt64();
        if (id >= vocabularySize ||
            id > std::numeric_limits<std::uint32_t>::max()) {
            throw MetadataError("Qwen special token ID is outside the vocabulary");
        }
        result.push_back(static_cast<std::uint32_t>(id));
    };
    if (value->isArray()) {
        for (const auto& token : value->asArray()) append(token);
    } else {
        append(*value);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

QwenTokenizerMetadata loadTokenizer(const std::filesystem::path& root,
                                    std::size_t expectedVocabularySize) {
    const auto tokenizerPath = root / "tokenizer.json";
    const auto configPath = root / "tokenizer_config.json";
    if (!std::filesystem::is_regular_file(tokenizerPath) ||
        !std::filesystem::is_regular_file(configPath)) {
        throw MetadataError(
            "Qwen checkpoint requires tokenizer.json and tokenizer_config.json");
    }
    const auto tokenizer = models::metadata::parseJsonFile(tokenizerPath);
    const auto tokenizerConfig = models::metadata::parseJsonFile(configPath);
    const auto modelConfig = models::metadata::parseJsonFile(root / "config.json");
    QwenTokenizerMetadata result;
    result.declaredVocabularySize = expectedVocabularySize;
    const auto& model = tokenizer.require("model");
    if (const auto* type = model.find("type")) {
        result.tokenizerModel = type->asString();
    }
    const auto& vocabulary = model.require("vocab");
    std::uint64_t maximumId{};
    if (vocabulary.isObject()) {
        result.vocabularyEntries = vocabulary.asObject().size();
        for (const auto& [token, idValue] : vocabulary.asObject()) {
            (void)token;
            maximumId = std::max(maximumId, idValue.asUInt64());
        }
    } else if (vocabulary.isArray()) {
        result.vocabularyEntries = vocabulary.asArray().size();
        maximumId = result.vocabularyEntries == 0
            ? 0 : static_cast<std::uint64_t>(result.vocabularyEntries - 1);
    } else {
        throw MetadataError("Qwen tokenizer vocabulary has an unsupported representation");
    }
    if (const auto* added = tokenizer.find("added_tokens")) {
        for (const auto& entry : added->asArray()) {
            maximumId = std::max(maximumId, entry.require("id").asUInt64());
            ++result.addedTokens;
        }
    }
    if (result.vocabularyEntries == 0 || maximumId >= expectedVocabularySize) {
        throw MetadataError("Qwen tokenizer IDs disagree with model vocab_size");
    }
    if (const auto* tokenizerClass = tokenizerConfig.find("tokenizer_class")) {
        result.tokenizerClass = tokenizerClass->asString();
    }
    result.hasChatTemplate = tokenizerConfig.find("chat_template") != nullptr;
    result.hasSpecialTokensMap =
        std::filesystem::is_regular_file(root / "special_tokens_map.json");
    result.bosTokenIds = tokenIds(modelConfig, "bos_token_id", expectedVocabularySize);
    result.eosTokenIds = tokenIds(modelConfig, "eos_token_id", expectedVocabularySize);
    result.padTokenIds = tokenIds(modelConfig, "pad_token_id", expectedVocabularySize);
    result.validate();
    return result;
}

void writeIds(std::ostringstream& output, std::span<const std::uint32_t> ids) {
    output << '[';
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) output << ',';
        output << ids[index];
    }
    output << ']';
}

} // namespace

void QwenTokenizerMetadata::validate() const {
    if (declaredVocabularySize == 0 || vocabularyEntries == 0 ||
        vocabularyEntries > declaredVocabularySize) {
        throw std::invalid_argument("Qwen tokenizer metadata is inconsistent");
    }
    const auto validIds = [&](const std::vector<std::uint32_t>& ids) {
        return std::all_of(ids.begin(), ids.end(), [&](const auto id) {
            return id < declaredVocabularySize;
        });
    };
    if (!validIds(bosTokenIds) || !validIds(eosTokenIds) || !validIds(padTokenIds)) {
        throw std::invalid_argument("Qwen tokenizer special token is invalid");
    }
}

std::string QwenTokenizerMetadata::toJson() const {
    validate();
    std::ostringstream output;
    output << "{\n  \"schema\": \"hypermoe.qwen-tokenizer-metadata.v1\",\n"
           << "  \"tokenizer_class\": \"" << escape(tokenizerClass) << "\",\n"
           << "  \"tokenizer_model\": \"" << escape(tokenizerModel) << "\",\n"
           << "  \"declared_vocabulary_size\": " << declaredVocabularySize << ",\n"
           << "  \"vocabulary_entries\": " << vocabularyEntries << ",\n"
           << "  \"added_tokens\": " << addedTokens << ",\n"
           << "  \"bos_token_ids\": ";
    writeIds(output, bosTokenIds);
    output << ",\n  \"eos_token_ids\": ";
    writeIds(output, eosTokenIds);
    output << ",\n  \"pad_token_ids\": ";
    writeIds(output, padTokenIds);
    output << ",\n  \"has_chat_template\": "
           << (hasChatTemplate ? "true" : "false")
           << ",\n  \"has_special_tokens_map\": "
           << (hasSpecialTokensMap ? "true" : "false") << "\n}\n";
    return output.str();
}

QwenCheckpointDescriptor QwenCheckpointLoader::load(
    const std::filesystem::path& artifact) const {
    QwenCheckpointDescriptor result;
    result.root = rootFor(artifact);
    result.manifest = QwenImporter{}.inspect(artifact);
    if (!result.manifest.runtimeArchitecture ||
        result.manifest.runtimeArchitecture->vocabularySize == 0 ||
        !result.manifest.modelIO || result.manifest.layers.size() !=
            result.manifest.runtimeArchitecture->layerCount) {
        throw MetadataError("Qwen checkpoint lacks complete forward tensor mappings");
    }
    result.validation = validation::CheckpointValidator::validate(
        artifact, result.manifest);
    result.tokenizer = loadTokenizer(
        result.root, result.manifest.runtimeArchitecture->vocabularySize);
    return result;
}

} // namespace hypermoe::importer::qwen
