#include "importer/SafeTensors.hpp"

#include "importer/SafeTensorShardManager.hpp"

#include "models/metadata/JsonValue.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace hypermoe::importer {
namespace {

constexpr std::uint64_t maximumHeaderBytes = 100ULL * 1024ULL * 1024ULL;

tensor::DType parseSafeTensorDType(std::string_view value) {
    if (value == "F32") return tensor::DType::FP32;
    if (value == "F16") return tensor::DType::FP16;
    if (value == "BF16") return tensor::DType::BF16;
    if (value == "I8") return tensor::DType::INT8;
    throw models::metadata::MetadataError(
        "SafeTensors dtype is not yet supported by HyperMoE: " + std::string(value));
}

std::uint64_t littleEndian64(const std::array<unsigned char, 8>& bytes) noexcept {
    std::uint64_t value{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
}

bool safeRelative(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) return false;
    return std::none_of(path.begin(), path.end(), [](const auto& component) {
        return component == "..";
    });
}

} // namespace

std::vector<models::ManifestTensor> SafeTensors::inspectFile(
    const std::filesystem::path& file,
    const std::filesystem::path& artifactRoot) {
    std::error_code error;
    const auto canonicalFile = std::filesystem::canonical(file, error);
    if (error) throw models::metadata::MetadataError("cannot resolve SafeTensors file");
    const auto canonicalRoot = std::filesystem::canonical(artifactRoot, error);
    if (error) throw models::metadata::MetadataError("cannot resolve artifact root");
    auto relative = std::filesystem::relative(canonicalFile, canonicalRoot, error);
    if (error || !safeRelative(relative)) {
        throw models::metadata::MetadataError("SafeTensors file escapes artifact root");
    }
    const auto fileSize = std::filesystem::file_size(canonicalFile, error);
    if (error || fileSize < 8) {
        throw models::metadata::MetadataError("invalid SafeTensors file size");
    }
    std::ifstream input(canonicalFile, std::ios::binary);
    std::array<unsigned char, 8> headerSizeBytes{};
    input.read(reinterpret_cast<char*>(headerSizeBytes.data()),
               static_cast<std::streamsize>(headerSizeBytes.size()));
    if (!input) throw models::metadata::MetadataError("cannot read SafeTensors header size");
    const auto headerSize = littleEndian64(headerSizeBytes);
    if (headerSize == 0 || headerSize > maximumHeaderBytes ||
        headerSize > fileSize - 8 ||
        headerSize > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw models::metadata::MetadataError("invalid SafeTensors header length");
    }
    std::string header(static_cast<std::size_t>(headerSize), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) throw models::metadata::MetadataError("truncated SafeTensors header");
    const auto root = models::metadata::parseJson(header);
    std::vector<models::ManifestTensor> tensors;
    for (const auto& [name, value] : root.asObject()) {
        if (name == "__metadata__") continue;
        std::vector<std::size_t> dimensions;
        for (const auto& dimension : value.require("shape").asArray()) {
            const auto number = dimension.asUInt64();
            if (number == 0 || number > std::numeric_limits<std::size_t>::max()) {
                throw models::metadata::MetadataError("invalid SafeTensors shape");
            }
            dimensions.push_back(static_cast<std::size_t>(number));
        }
        if (dimensions.empty()) {
            throw models::metadata::MetadataError("scalar SafeTensors are not supported");
        }
        const auto& offsets = value.require("data_offsets").asArray();
        if (offsets.size() != 2) {
            throw models::metadata::MetadataError("SafeTensors range needs two offsets");
        }
        const auto start = offsets[0].asUInt64();
        const auto end = offsets[1].asUInt64();
        if (end <= start) {
            throw models::metadata::MetadataError("SafeTensors range is empty or reversed");
        }
        const auto dataBase = 8ULL + headerSize;
        if (start > std::numeric_limits<std::uint64_t>::max() - dataBase ||
            end > fileSize - dataBase) {
            throw models::metadata::MetadataError("SafeTensors range exceeds file");
        }
        models::ManifestTensor tensor;
        tensor.name = name;
        tensor.sourceFile = relative.lexically_normal();
        tensor.offset = dataBase + start;
        tensor.size = end - start;
        tensor.dtype = parseSafeTensorDType(value.require("dtype").asString());
        tensor.shape = tensor::Shape(std::move(dimensions));
        if (tensor.shape.storageElementCount() >
                std::numeric_limits<std::uint64_t>::max() /
                    tensor::sizeOf(tensor.dtype) ||
            static_cast<std::uint64_t>(tensor.shape.storageElementCount()) *
                    tensor::sizeOf(tensor.dtype) != tensor.size) {
            throw models::metadata::MetadataError(
                "SafeTensors shape, dtype, and byte range disagree");
        }
        tensors.push_back(std::move(tensor));
    }
    auto ranges = tensors;
    std::sort(ranges.begin(), ranges.end(), [](const auto& left, const auto& right) {
        return left.offset < right.offset;
    });
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index - 1].offset + ranges[index - 1].size > ranges[index].offset) {
            throw models::metadata::MetadataError(
                "SafeTensors tensor byte ranges overlap");
        }
    }
    return tensors;
}

std::vector<models::ManifestTensor> SafeTensors::inspectArtifact(
    const std::filesystem::path& artifact) {
    const SafeTensorShardManager shards(artifact);
    return {shards.tensors().begin(), shards.tensors().end()};
}

} // namespace hypermoe::importer
