#pragma once

#include "importer/qwen/QwenCheckpointLoader.hpp"
#include "tools/model_convert/ExpertPacker.hpp"

#include <chrono>
#include <filesystem>
#include <string>

namespace hypermoe::conversion::real_checkpoint {

struct RealCheckpointConversionReport {
    std::string modelName;
    std::string sourceArchitecture;
    importer::qwen::QwenTokenizerMetadata tokenizer;
    importer::validation::CheckpointValidationReport checkpoint;
    PackingReport packing;
    std::chrono::nanoseconds elapsed{};

    [[nodiscard]] std::string toJson() const;
};

class RealCheckpointConverter {
public:
    [[nodiscard]] RealCheckpointConversionReport convertQwen(
        const std::filesystem::path& checkpoint,
        const std::filesystem::path& outputDirectory,
        ExpertPackingOptions packingOptions = {}) const;
};

} // namespace hypermoe::conversion::real_checkpoint
