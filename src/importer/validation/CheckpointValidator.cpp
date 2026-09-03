#include "importer/validation/CheckpointValidator.hpp"

#include "importer/SafeTensorShardManager.hpp"

#include <limits>
#include <set>
#include <stdexcept>

namespace hypermoe::importer::validation {

CheckpointValidationReport CheckpointValidator::validate(
    const std::filesystem::path& artifact,
    const models::ModelManifest& manifest) {
    manifest.validate();
    SafeTensorShardManager shards(artifact);
    CheckpointValidationReport report;
    report.shardCount = shards.shards().size();
    report.tensorCount = shards.tensors().size();
    report.expertCount = manifest.experts.size();
    report.routerTensorCount = manifest.router.tensors.size();
    std::set<std::string> referenced;
    for (const auto& value : manifest.tensors) {
        const auto* source = shards.find(value.name);
        if (!source || source->sourceFile != value.sourceFile ||
            source->offset != value.offset || source->size != value.size ||
            source->dtype != value.dtype || source->shape != value.shape) {
            throw std::invalid_argument(
                "manifest tensor does not match source checkpoint: " + value.name);
        }
        if (report.referencedBytes >
            std::numeric_limits<std::uint64_t>::max() - value.size) {
            throw std::overflow_error("checkpoint referenced byte count overflows");
        }
        report.referencedBytes += value.size;
        referenced.insert(value.name);
        ++report.dtypeCounts[std::string(tensor::toString(value.dtype))];
    }
    report.referencedTensorCount = referenced.size();
    return report;
}

} // namespace hypermoe::importer::validation
