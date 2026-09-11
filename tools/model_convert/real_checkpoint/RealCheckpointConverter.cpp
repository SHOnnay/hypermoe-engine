#include "tools/model_convert/real_checkpoint/RealCheckpointConverter.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace hypermoe::conversion::real_checkpoint {
namespace {

std::string escape(std::string_view value) {
    std::string result;
    for (const auto character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

void writeFile(const std::filesystem::path& path, std::string_view contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) throw std::runtime_error("cannot write checkpoint conversion metadata");
}

} // namespace

std::string RealCheckpointConversionReport::toJson() const {
    std::ostringstream output;
    output << "{\n  \"schema\": \"hypermoe.real-checkpoint-conversion.v1\",\n"
           << "  \"model\": \"" << escape(modelName) << "\",\n"
           << "  \"source_architecture\": \"" << escape(sourceArchitecture)
           << "\",\n  \"layers\": " << packing.layers
           << ",\n  \"experts\": " << packing.experts
           << ",\n  \"parameters\": " << checkpoint.totalParameters
           << ",\n  \"source_shards\": " << checkpoint.shardCount
           << ",\n  \"source_tensors\": " << checkpoint.tensorCount
           << ",\n  \"packed_bytes\": " << packing.bytesWritten
           << ",\n  \"elapsed_ms\": "
           << std::chrono::duration<double, std::milli>(elapsed).count()
           << ",\n  \"tokenizer\": " << tokenizer.toJson() << "}\n";
    return output.str();
}

RealCheckpointConversionReport RealCheckpointConverter::convertQwen(
    const std::filesystem::path& checkpoint,
    const std::filesystem::path& outputDirectory) const {
    const auto started = std::chrono::steady_clock::now();
    const auto descriptor = importer::qwen::QwenCheckpointLoader{}.load(checkpoint);
    RealCheckpointConversionReport result;
    result.modelName = descriptor.manifest.modelName;
    result.sourceArchitecture = descriptor.manifest.sourceArchitecture;
    result.tokenizer = descriptor.tokenizer;
    result.checkpoint = descriptor.validation;
    bool packed{};
    try {
        result.packing = ExpertPacker{}.pack(
            descriptor.manifest, descriptor.root, outputDirectory);
        packed = true;
        if (!result.packing.validationPassed) {
            throw std::runtime_error("real checkpoint packing did not pass validation");
        }
        result.elapsed = std::chrono::steady_clock::now() - started;
        writeFile(outputDirectory / "tokenizer_metadata.json",
                  result.tokenizer.toJson());
        for (const auto* filename : {"tokenizer.json", "tokenizer_config.json",
                                     "special_tokens_map.json"}) {
            const auto source = descriptor.root / filename;
            if (std::filesystem::is_regular_file(source)) {
                std::filesystem::copy_file(
                    source, outputDirectory / filename,
                    std::filesystem::copy_options::none);
            }
        }
        writeFile(outputDirectory / "real_checkpoint_report.json", result.toJson());
        return result;
    } catch (...) {
        if (packed) {
            std::error_code error;
            std::filesystem::remove_all(outputDirectory, error);
        }
        throw;
    }
}

} // namespace hypermoe::conversion::real_checkpoint
