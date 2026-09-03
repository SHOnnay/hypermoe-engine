#include "storage/ExpertIndex.hpp"

#include "tensor/DType.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace hypermoe::storage {
namespace {

constexpr std::array<char, 8> kMagic{'H', 'M', 'O', 'E', 'I', 'D', 'X', '\0'};
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::size_t kHeaderSize = 32;

template <typename T>
void writeLittleEndian(std::ostream& output, T value) {
    static_assert(std::is_unsigned_v<T>);
    std::array<char, sizeof(T)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<char>((value >> (index * 8U)) & static_cast<T>(0xffU));
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
T readLittleEndian(std::istream& input) {
    static_assert(std::is_unsigned_v<T>);
    std::array<unsigned char, sizeof(T)> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw StorageError("truncated expert index");
    }
    T value{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<T>(bytes[index]) << (index * 8U);
    }
    return value;
}

void ensureWrite(const std::ostream& output, const std::filesystem::path& path) {
    if (!output) {
        throw StorageError("failed writing expert index: " + path.string());
    }
}

} // namespace

ExpertIndex::ExpertIndex(std::vector<ExpertRecord> records)
    : records_(std::move(records)) {
    rebuildLookup();
}

ExpertIndex::ExpertIndex(std::vector<ExpertRecord> records,
                         std::vector<ProjectionRecord> projections)
    : records_(std::move(records)), projections_(std::move(projections)) {
    rebuildLookup();
}

ExpertIndex ExpertIndex::load(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw StorageError("cannot open expert index: " + path.string());
    }

    std::array<char, kMagic.size()> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != kMagic) {
        throw StorageError("invalid expert index magic: " + path.string());
    }
    const auto version = readLittleEndian<std::uint32_t>(input);
    const auto endian = readLittleEndian<std::uint32_t>(input);
    const auto recordSize = readLittleEndian<std::uint32_t>(input);
    const auto extensionRecordSize = readLittleEndian<std::uint32_t>(input);
    const auto count = readLittleEndian<std::uint64_t>(input);
    if (version != legacyFormatVersion && version != formatVersion) {
        throw StorageError("unsupported expert index version");
    }
    if (endian != kEndianMarker) {
        throw StorageError("invalid expert index endian marker");
    }
    if (recordSize != serializedRecordSize ||
        (version == legacyFormatVersion && extensionRecordSize != 0) ||
        (version == formatVersion &&
         extensionRecordSize != serializedProjectionRecordSize)) {
        throw StorageError("unsupported expert index record size");
    }
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) /
                    serializedRecordSize) {
        throw StorageError("expert index record count is too large");
    }
    std::error_code sizeError;
    const auto fileSize = std::filesystem::file_size(path, sizeError);
    if (sizeError || count > (std::numeric_limits<std::uint64_t>::max() - kHeaderSize) /
                                 serializedRecordSize) {
        throw StorageError("expert index size does not match its record count");
    }
    const auto baseSize = kHeaderSize + count * serializedRecordSize;
    if (fileSize < baseSize ||
        (version == legacyFormatVersion && fileSize != baseSize)) {
        throw StorageError("expert index size does not match its record count");
    }

    std::vector<ExpertRecord> records;
    records.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        records.push_back({readLittleEndian<std::uint32_t>(input),
                           readLittleEndian<std::uint32_t>(input),
                           readLittleEndian<std::uint64_t>(input),
                           readLittleEndian<std::uint64_t>(input),
                           readLittleEndian<std::uint32_t>(input),
                           readLittleEndian<std::uint32_t>(input)});
    }
    std::vector<ProjectionRecord> projections;
    if (version == formatVersion) {
        const auto projectionCount = readLittleEndian<std::uint64_t>(input);
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        if (baseSize > maximum - 8) {
            throw StorageError("expert projection index size is invalid");
        }
        const auto projectionBase = baseSize + 8;
        if (projectionCount >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            projectionCount >
                (maximum - projectionBase) /
                    serializedProjectionRecordSize ||
            fileSize != projectionBase +
                            projectionCount * serializedProjectionRecordSize) {
            throw StorageError("expert projection index size is invalid");
        }
        projections.reserve(static_cast<std::size_t>(projectionCount));
        for (std::uint64_t index = 0; index < projectionCount; ++index) {
            ProjectionRecord record;
            record.layer_id = readLittleEndian<std::uint32_t>(input);
            record.expert_id = readLittleEndian<std::uint32_t>(input);
            record.projection_type = static_cast<ProjectionType>(
                readLittleEndian<std::uint32_t>(input));
            record.dtype = readLittleEndian<std::uint32_t>(input);
            record.rank = readLittleEndian<std::uint32_t>(input);
            record.reserved = readLittleEndian<std::uint32_t>(input);
            record.offset = readLittleEndian<std::uint64_t>(input);
            record.size = readLittleEndian<std::uint64_t>(input);
            for (auto& dimension : record.shape) {
                dimension = readLittleEndian<std::uint64_t>(input);
            }
            record.checksum = readLittleEndian<std::uint32_t>(input);
            record.trailing_reserved = readLittleEndian<std::uint32_t>(input);
            projections.push_back(record);
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw StorageError("expert index contains trailing data");
    }
    return ExpertIndex(std::move(records), std::move(projections));
}

void ExpertIndex::save(const std::filesystem::path& path) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw StorageError("cannot create expert index: " + path.string());
    }
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    const auto version = projections_.empty() ? legacyFormatVersion : formatVersion;
    writeLittleEndian(output, version);
    writeLittleEndian(output, kEndianMarker);
    writeLittleEndian(output, static_cast<std::uint32_t>(serializedRecordSize));
    writeLittleEndian(output, projections_.empty()
                                  ? std::uint32_t{0}
                                  : static_cast<std::uint32_t>(serializedProjectionRecordSize));
    writeLittleEndian(output, static_cast<std::uint64_t>(records_.size()));
    static_assert(kHeaderSize == 8 + 4 + 4 + 4 + 4 + 8);
    for (const auto& record : records_) {
        writeLittleEndian(output, record.layer_id);
        writeLittleEndian(output, record.expert_id);
        writeLittleEndian(output, record.offset);
        writeLittleEndian(output, record.size);
        writeLittleEndian(output, record.quantization_type);
        writeLittleEndian(output, record.checksum);
    }
    if (!projections_.empty()) {
        writeLittleEndian(output, static_cast<std::uint64_t>(projections_.size()));
        for (const auto& record : projections_) {
            writeLittleEndian(output, record.layer_id);
            writeLittleEndian(output, record.expert_id);
            writeLittleEndian(output, static_cast<std::uint32_t>(record.projection_type));
            writeLittleEndian(output, record.dtype);
            writeLittleEndian(output, record.rank);
            writeLittleEndian(output, record.reserved);
            writeLittleEndian(output, record.offset);
            writeLittleEndian(output, record.size);
            for (const auto dimension : record.shape) writeLittleEndian(output, dimension);
            writeLittleEndian(output, record.checksum);
            writeLittleEndian(output, record.trailing_reserved);
        }
    }
    ensureWrite(output, path);
}

std::optional<ExpertRecord>
ExpertIndex::find(std::uint32_t layerId, std::uint32_t expertId) const noexcept {
    const auto it = lookup_.find(key(layerId, expertId));
    return it == lookup_.end() ? std::nullopt
                               : std::optional<ExpertRecord>(records_[it->second]);
}

std::span<const ExpertRecord> ExpertIndex::records() const noexcept {
    return records_;
}

std::span<const ProjectionRecord> ExpertIndex::projections() const noexcept {
    return projections_;
}

std::optional<ProjectionRecord> ExpertIndex::findProjection(
    std::uint32_t layerId,
    std::uint32_t expertId,
    ProjectionType type) const noexcept {
    const auto found = projectionLookup_.find(key(layerId, expertId));
    const auto typeIndex = static_cast<std::size_t>(type);
    if (found == projectionLookup_.end() || typeIndex >= found->second.size() ||
        found->second[typeIndex] == std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    return projections_[found->second[typeIndex]];
}

std::size_t ExpertIndex::size() const noexcept {
    return records_.size();
}

std::uint64_t ExpertIndex::key(std::uint32_t layerId, std::uint32_t expertId) noexcept {
    return (static_cast<std::uint64_t>(layerId) << 32U) |
           static_cast<std::uint64_t>(expertId);
}

void ExpertIndex::rebuildLookup() {
    lookup_.reserve(records_.size());
    for (std::size_t index = 0; index < records_.size(); ++index) {
        const auto& record = records_[index];
        if (record.size == 0) {
            throw StorageError("expert index contains a zero-sized record");
        }
        if (record.offset > std::numeric_limits<std::uint64_t>::max() - record.size) {
            throw StorageError("expert index record range overflows");
        }
        if (!lookup_.emplace(key(record.layer_id, record.expert_id), index).second) {
            throw StorageError("expert index contains a duplicate layer/expert key");
        }
    }
    projectionLookup_.reserve(records_.size());
    for (std::size_t projectionIndex = 0; projectionIndex < projections_.size();
         ++projectionIndex) {
        const auto& projection = projections_[projectionIndex];
        if (projection.size == 0 || projection.rank == 0 || projection.rank > 4 ||
            projection.dtype > static_cast<std::uint32_t>(tensor::DType::INT8) ||
            projection.reserved != 0 || projection.trailing_reserved != 0 ||
            static_cast<std::uint32_t>(projection.projection_type) >
                static_cast<std::uint32_t>(ProjectionType::Down)) {
            throw StorageError("expert index contains an invalid projection record");
        }
        const auto expert = find(projection.layer_id, projection.expert_id);
        if (!expert || projection.offset < expert->offset ||
            projection.offset > std::numeric_limits<std::uint64_t>::max() -
                                    projection.size ||
            projection.offset + projection.size > expert->offset + expert->size) {
            throw StorageError("projection record escapes its expert payload");
        }
        const auto identity = key(projection.layer_id, projection.expert_id);
        auto [found, inserted] = projectionLookup_.try_emplace(
            identity,
            std::array<std::size_t, 3>{std::numeric_limits<std::size_t>::max(),
                                       std::numeric_limits<std::size_t>::max(),
                                       std::numeric_limits<std::size_t>::max()});
        (void)inserted;
        auto& position = found->second[static_cast<std::size_t>(
            projection.projection_type)];
        if (position != std::numeric_limits<std::size_t>::max()) {
            throw StorageError("expert index contains a duplicate projection record");
        }
        position = projectionIndex;
        std::uint64_t elements{1};
        for (std::uint32_t index = 0; index < projection.rank; ++index) {
            if (projection.shape[index] == 0) {
                throw StorageError("projection record contains a zero dimension");
            }
            if (elements > std::numeric_limits<std::uint64_t>::max() /
                               projection.shape[index]) {
                throw StorageError("projection shape size overflows");
            }
            elements *= projection.shape[index];
        }
        for (std::uint32_t index = projection.rank; index < projection.shape.size();
             ++index) {
            if (projection.shape[index] != 0) {
                throw StorageError("projection has dimensions beyond its declared rank");
            }
        }
        const auto elementBytes = tensor::sizeOf(
            static_cast<tensor::DType>(projection.dtype));
        if (elements > std::numeric_limits<std::uint64_t>::max() / elementBytes ||
            elements * elementBytes != projection.size) {
            throw StorageError("projection shape and dtype do not match its size");
        }
    }
}

} // namespace hypermoe::storage
