#include "tools/model_convert/ExpertPacker.hpp"

#include "hypermoe/experts/expert.hpp"
#include "importer/validation/CheckpointValidator.hpp"
#include "storage/ExpertIndex.hpp"
#include "storage/ExpertStore.hpp"
#include "tools/model_convert/WeightConverter.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hypermoe::conversion {
namespace {

constexpr std::uint64_t alignment = 4096;

std::uint64_t alignUp(std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
        throw storage::StorageError("packed expert offset overflow");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

std::vector<std::byte> readRange(const std::filesystem::path& root,
                                 const models::ManifestTensor& tensor,
                                 const models::ProjectionLocation& projection) {
    const auto path = root / tensor.sourceFile;
    std::ifstream input(path, std::ios::binary);
    if (!input || projection.offset >
                      static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        projection.size >
            static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
        projection.size >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw storage::StorageError("cannot read source projection range");
    }
    input.seekg(static_cast<std::streamoff>(projection.offset));
    std::vector<std::byte> bytes(static_cast<std::size_t>(projection.size));
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!input) throw storage::StorageError("source projection range is truncated");
    return bytes;
}

std::uint32_t quantizationFor(tensor::DType dtype) {
    switch (dtype) {
    case tensor::DType::FP32: return static_cast<std::uint32_t>(QuantizationType::Fp32);
    case tensor::DType::FP16: return static_cast<std::uint32_t>(QuantizationType::Fp16);
    case tensor::DType::BF16: return static_cast<std::uint32_t>(QuantizationType::Bf16);
    case tensor::DType::INT8: return static_cast<std::uint32_t>(QuantizationType::Int8);
    }
    throw storage::StorageError("unsupported packed expert dtype");
}

std::array<std::uint64_t, 4> shapeArray(const tensor::Shape& shape) {
    if (shape.rank() > 4) throw storage::StorageError("projection rank exceeds index limit");
    std::array<std::uint64_t, 4> result{};
    std::copy(shape.dimensions().begin(), shape.dimensions().end(), result.begin());
    return result;
}

} // namespace

std::string PackingReport::toJson() const {
    std::ostringstream output;
    output << "{\n  \"schema\": \"hypermoe.conversion-report.v1\",\n"
           << "  \"layers\": " << layers << ",\n"
           << "  \"experts\": " << experts << ",\n"
           << "  \"projections\": " << projections << ",\n"
           << "  \"source_tensors\": " << sourceTensors << ",\n"
           << "  \"parameters_indexed\": " << parametersIndexed << ",\n"
           << "  \"bytes_read\": " << bytesRead << ",\n"
           << "  \"bytes_written\": " << bytesWritten << ",\n"
           << "  \"shards\": " << shardCount << ",\n"
           << "  \"validation_passed\": "
           << (validationPassed ? "true" : "false") << ",\n"
           << "  \"dtype_tensors\": {";
    bool first = true;
    for (const auto& [dtype, count] : dtypeTensors) {
        if (!first) output << ',';
        output << "\n    \"" << dtype << "\": " << count;
        first = false;
    }
    if (!dtypeTensors.empty()) output << '\n' << "  ";
    output << "}\n}\n";
    return output.str();
}

PackingReport ExpertPacker::pack(const models::ModelManifest& sourceManifest,
                                 const std::filesystem::path& artifactRoot,
                                 const std::filesystem::path& outputDirectory) const {
    sourceManifest.validate();
    const auto checkpoint = importer::validation::CheckpointValidator::validate(
        artifactRoot, sourceManifest);
    if (std::filesystem::exists(outputDirectory)) {
        throw storage::StorageError("refusing to overwrite model conversion output");
    }
    std::filesystem::create_directories(outputDirectory);
    try {
        std::ofstream output(outputDirectory / "experts.bin",
                             std::ios::binary | std::ios::trunc);
        if (!output) throw storage::StorageError("cannot create packed expert data");
        models::ModelManifest packed = sourceManifest;
        packed.tensors.clear();
        packed.experts.clear();
        packed.router.tensors.clear();
        packed.router.layout = models::TensorLayout::InputOutput;
        std::vector<storage::ExpertRecord> experts;
        std::vector<storage::ProjectionRecord> projections;
        PackingReport report;
        report.layers = sourceManifest.config.layerCount;
        report.sourceTensors = sourceManifest.tensors.size();
        report.shardCount = checkpoint.shardCount;
        for (const auto& sourceTensor : sourceManifest.tensors) {
            if (report.parametersIndexed >
                std::numeric_limits<std::uint64_t>::max() -
                    sourceTensor.shape.elementCount()) {
                throw storage::StorageError("conversion parameter count overflow");
            }
            report.parametersIndexed += sourceTensor.shape.elementCount();
            ++report.dtypeTensors[std::string(tensor::toString(sourceTensor.dtype))];
        }
        std::uint64_t cursor{};
        const std::array<char, alignment> zeros{};

        for (const auto& sourceExpert : sourceManifest.experts) {
            const auto expertStart = alignUp(cursor);
            if (expertStart != cursor) {
                output.write(zeros.data(), static_cast<std::streamsize>(expertStart - cursor));
            }
            cursor = expertStart;
            std::vector<std::byte> expertBytes;
            const auto packProjection = [&](const models::ProjectionLocation& location,
                                            storage::ProjectionType type,
                                            std::string role) {
                const auto* tensor = sourceManifest.findTensor(location.tensorName);
                if (!tensor) throw storage::StorageError("projection source is missing");
                auto bytes = readRange(artifactRoot, *tensor, location);
                report.bytesRead += bytes.size();
                auto converted = WeightConverter::convert(
                    bytes, location.shape, tensor->dtype, location.layout);
                if (expertBytes.size() >
                    std::numeric_limits<std::uint64_t>::max() - expertStart) {
                    throw storage::StorageError("packed projection offset overflow");
                }
                const auto projectionOffset = expertStart + expertBytes.size();
                expertBytes.insert(expertBytes.end(), converted.bytes.begin(),
                                   converted.bytes.end());
                const auto name = "layers." + std::to_string(sourceExpert.layerId) +
                                  ".experts." + std::to_string(sourceExpert.expertId) +
                                  "." + role;
                packed.tensors.push_back({name, "experts.bin", projectionOffset,
                                          static_cast<std::uint64_t>(converted.bytes.size()),
                                          converted.dtype, converted.shape});
                models::ProjectionLocation packedLocation{
                    name, projectionOffset,
                    static_cast<std::uint64_t>(converted.bytes.size()), converted.shape,
                    models::TensorLayout::InputOutput};
                projections.push_back({sourceExpert.layerId, sourceExpert.expertId,
                    type, static_cast<std::uint32_t>(converted.dtype),
                    static_cast<std::uint32_t>(converted.shape.rank()), 0,
                    projectionOffset,
                    static_cast<std::uint64_t>(converted.bytes.size()),
                    shapeArray(converted.shape), storage::ExpertStore::crc32(converted.bytes), 0});
                ++report.projections;
                return packedLocation;
            };
            models::ManifestExpertMapping mapping;
            mapping.layerId = sourceExpert.layerId;
            mapping.expertId = sourceExpert.expertId;
            const auto* gateTensor = sourceManifest.findTensor(sourceExpert.gate.tensorName);
            const auto* upTensor = sourceManifest.findTensor(sourceExpert.up.tensorName);
            const auto* downTensor = sourceManifest.findTensor(sourceExpert.down.tensorName);
            if (!gateTensor || !upTensor || !downTensor ||
                gateTensor->dtype != upTensor->dtype ||
                gateTensor->dtype != downTensor->dtype) {
                throw storage::StorageError(
                    "expert projections must exist and use one storage dtype");
            }
            const auto dtype = gateTensor->dtype;
            mapping.gate = packProjection(sourceExpert.gate, storage::ProjectionType::Gate, "gate");
            mapping.up = packProjection(sourceExpert.up, storage::ProjectionType::Up, "up");
            mapping.down = packProjection(sourceExpert.down, storage::ProjectionType::Down, "down");
            packed.experts.push_back(std::move(mapping));
            if (expertBytes.size() >
                static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                throw storage::StorageError("packed expert is too large for stream I/O");
            }
            output.write(reinterpret_cast<const char*>(expertBytes.data()),
                         static_cast<std::streamsize>(expertBytes.size()));
            if (!output) throw storage::StorageError("failed writing packed expert");
            const auto expertSize = static_cast<std::uint64_t>(expertBytes.size());
            if (expertStart > std::numeric_limits<std::uint64_t>::max() - expertSize) {
                throw storage::StorageError("packed expert range overflow");
            }
            cursor = expertStart + expertSize;
            experts.push_back({sourceExpert.layerId, sourceExpert.expertId, expertStart,
                               expertSize, quantizationFor(dtype),
                               storage::ExpertStore::crc32(expertBytes)});
            ++report.experts;
        }

        for (const auto& sourceRouter : sourceManifest.router.tensors) {
            const auto* tensor = sourceManifest.findTensor(sourceRouter.tensorName);
            if (!tensor) throw storage::StorageError("router source is missing");
            models::ProjectionLocation location{tensor->name, tensor->offset, tensor->size,
                                                tensor->shape, sourceManifest.router.layout};
            auto bytes = readRange(artifactRoot, *tensor, location);
            report.bytesRead += bytes.size();
            auto converted = WeightConverter::convert(bytes, tensor->shape, tensor->dtype,
                                                       sourceManifest.router.layout);
            const auto routerOffset = cursor;
            if (converted.bytes.size() >
                    static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
                converted.bytes.size() >
                    std::numeric_limits<std::uint64_t>::max() - cursor) {
                throw storage::StorageError("packed router range is too large");
            }
            output.write(reinterpret_cast<const char*>(converted.bytes.data()),
                         static_cast<std::streamsize>(converted.bytes.size()));
            if (!output) throw storage::StorageError("failed writing packed router");
            cursor += converted.bytes.size();
            const auto name = "layers." + std::to_string(sourceRouter.layerId) + ".router";
            packed.tensors.push_back({name, "experts.bin", routerOffset,
                                      static_cast<std::uint64_t>(converted.bytes.size()),
                                      converted.dtype, converted.shape});
            packed.router.tensors.push_back({sourceRouter.layerId, name});
        }
        output.close();
        if (!output) throw storage::StorageError("failed closing packed expert data");
        storage::ExpertIndex(std::move(experts), std::move(projections))
            .save(outputDirectory / "experts.index");
        packed.validate();
        packed.save(outputDirectory / "manifest.json");
        report.bytesWritten = cursor;
        {
            storage::ExpertStore verification(outputDirectory);
            for (const auto& record : verification.index().records()) {
                (void)verification.mappedExpert(record.layer_id, record.expert_id, true);
            }
            for (const auto& projection : verification.index().projections()) {
                (void)verification.mappedProjection(
                    projection.layer_id, projection.expert_id,
                    projection.projection_type, true);
            }
        }
        report.validationPassed = true;
        std::ofstream reportFile(outputDirectory / "conversion_report.json",
                                 std::ios::binary | std::ios::trunc);
        const auto reportJson = report.toJson();
        reportFile.write(reportJson.data(),
                         static_cast<std::streamsize>(reportJson.size()));
        if (!reportFile) {
            throw storage::StorageError("failed writing conversion report");
        }
        return report;
    } catch (...) {
        std::error_code error;
        std::filesystem::remove_all(outputDirectory, error);
        throw;
    }
}

} // namespace hypermoe::conversion
