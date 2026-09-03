#include "storage/ExpertStore.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

namespace hypermoe::storage {
namespace {

constexpr std::uint64_t kAlignment = 4096;

std::uint64_t alignUp(std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - (kAlignment - 1)) {
        throw StorageError("expert data offset overflow");
    }
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw StorageError("cannot open model metadata: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string readStoreMetadata(const std::filesystem::path& directory) {
    const auto legacy = directory / "metadata.json";
    if (std::filesystem::exists(legacy)) return readTextFile(legacy);
    return readTextFile(directory / "manifest.json");
}

} // namespace

ExpertStore::ExpertStore(const std::filesystem::path& modelDirectory)
    : dataPath_(modelDirectory / "experts.bin"),
      index_(ExpertIndex::load(modelDirectory / "experts.index")),
      mappedData_(dataPath_),
      metadataJson_(readStoreMetadata(modelDirectory)) {
    if (metadataJson_.empty()) {
        throw StorageError("model metadata manifest is empty");
    }
    for (const auto& record : index_.records()) {
        if (record.offset > mappedData_.size() ||
            record.size > static_cast<std::uint64_t>(mappedData_.size()) - record.offset) {
            throw StorageError("expert index references data outside experts.bin");
        }
    }
}

std::span<const std::byte> ExpertStore::mappedProjection(
    std::uint32_t layerId,
    std::uint32_t expertId,
    ProjectionType type,
    bool validateChecksum) const {
    const auto record = index_.findProjection(layerId, expertId, type);
    if (!record) throw StorageError("expert projection is not present in the index");
    const auto bytes = mappedData_.view(record->offset, record->size);
    if (validateChecksum && crc32(bytes) != record->checksum) {
        throw StorageError("expert projection checksum mismatch: data is corrupted");
    }
    return bytes;
}

void ExpertStore::create(const std::filesystem::path& modelDirectory,
                         std::span<const ExpertBlob> experts,
                         std::string_view metadataJson) {
    if (experts.empty()) {
        throw StorageError("cannot create an expert store without experts");
    }
    if (metadataJson.empty()) {
        throw StorageError("metadata JSON must not be empty");
    }
    if (std::filesystem::exists(modelDirectory / "experts.bin") ||
        std::filesystem::exists(modelDirectory / "experts.index") ||
        std::filesystem::exists(modelDirectory / "metadata.json")) {
        throw StorageError("refusing to overwrite an existing expert store");
    }
    std::filesystem::create_directories(modelDirectory);

    const auto dataPath = modelDirectory / "experts.bin";
    std::ofstream data(dataPath, std::ios::binary | std::ios::trunc);
    if (!data) {
        throw StorageError("cannot create expert data file: " + dataPath.string());
    }

    std::vector<ExpertRecord> records;
    records.reserve(experts.size());
    std::uint64_t cursor = 0;
    const std::array<char, kAlignment> zeroPage{};
    for (const auto& expert : experts) {
        if (expert.data.empty()) {
            throw StorageError("cannot store an empty expert");
        }
        const auto offset = alignUp(cursor);
        const auto padding = offset - cursor;
        if (padding != 0) {
            data.write(zeroPage.data(), static_cast<std::streamsize>(padding));
        }
        if (expert.data.size() > static_cast<std::size_t>(
                                     std::numeric_limits<std::streamsize>::max())) {
            throw StorageError("expert is too large for stream I/O");
        }
        data.write(reinterpret_cast<const char*>(expert.data.data()),
                   static_cast<std::streamsize>(expert.data.size()));
        if (!data) {
            throw StorageError("failed writing expert data");
        }
        records.push_back({expert.layerId,
                           expert.expertId,
                           offset,
                           static_cast<std::uint64_t>(expert.data.size()),
                           expert.quantizationType,
                           crc32(expert.data)});
        cursor = offset + static_cast<std::uint64_t>(expert.data.size());
    }
    data.close();
    if (!data) {
        throw StorageError("failed closing expert data file");
    }

    ExpertIndex(std::move(records)).save(modelDirectory / "experts.index");
    std::ofstream metadata(modelDirectory / "metadata.json",
                           std::ios::binary | std::ios::trunc);
    metadata.write(metadataJson.data(), static_cast<std::streamsize>(metadataJson.size()));
    if (!metadata) {
        throw StorageError("failed writing metadata.json");
    }
}

const ExpertIndex& ExpertStore::index() const noexcept {
    return index_;
}

const std::string& ExpertStore::metadataJson() const noexcept {
    return metadataJson_;
}

const std::filesystem::path& ExpertStore::dataPath() const noexcept {
    return dataPath_;
}

std::span<const std::byte>
ExpertStore::mappedExpert(std::uint32_t layerId,
                          std::uint32_t expertId,
                          bool validateChecksum) const {
    const auto record = requireRecord(layerId, expertId);
    const auto bytes = mappedData_.view(record.offset, record.size);
    if (validateChecksum) {
        validate(bytes, record);
    }
    return bytes;
}

std::vector<std::byte>
ExpertStore::readExpert(std::uint32_t layerId,
                        std::uint32_t expertId,
                        bool validateChecksum) const {
    const auto record = requireRecord(layerId, expertId);
    std::ifstream input(dataPath_, std::ios::binary);
    if (!input) {
        throw StorageError("cannot open expert data file for range read");
    }
    if (record.offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        record.size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw StorageError("expert range is too large for stream I/O");
    }
    input.seekg(static_cast<std::streamoff>(record.offset));
    std::vector<std::byte> bytes(static_cast<std::size_t>(record.size));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw StorageError("short range read from experts.bin");
    }
    if (validateChecksum) {
        validate(bytes, record);
    }
    return bytes;
}

std::uint32_t ExpertStore::crc32(std::span<const std::byte> bytes) noexcept {
    std::uint32_t crc = 0xffffffffU;
    for (const auto byte : bytes) {
        crc ^= static_cast<std::uint32_t>(std::to_integer<unsigned char>(byte));
        for (int bit = 0; bit < 8; ++bit) {
            const auto mask = 0U - (crc & 1U);
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

ExpertRecord ExpertStore::requireRecord(std::uint32_t layerId,
                                        std::uint32_t expertId) const {
    const auto record = index_.find(layerId, expertId);
    if (!record) {
        throw StorageError("expert is not present in the index");
    }
    return *record;
}

void ExpertStore::validate(std::span<const std::byte> bytes,
                           const ExpertRecord& record) {
    if (crc32(bytes) != record.checksum) {
        throw StorageError("expert checksum mismatch: data is corrupted");
    }
}

} // namespace hypermoe::storage
