#include "storage/ExpertIndex.hpp"

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
    (void)readLittleEndian<std::uint32_t>(input); // reserved
    const auto count = readLittleEndian<std::uint64_t>(input);
    if (version != formatVersion) {
        throw StorageError("unsupported expert index version");
    }
    if (endian != kEndianMarker) {
        throw StorageError("invalid expert index endian marker");
    }
    if (recordSize != serializedRecordSize) {
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
                                 serializedRecordSize ||
        fileSize != kHeaderSize + count * serializedRecordSize) {
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
    if (input.peek() != std::char_traits<char>::eof()) {
        throw StorageError("expert index contains trailing data");
    }
    return ExpertIndex(std::move(records));
}

void ExpertIndex::save(const std::filesystem::path& path) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw StorageError("cannot create expert index: " + path.string());
    }
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    writeLittleEndian(output, formatVersion);
    writeLittleEndian(output, kEndianMarker);
    writeLittleEndian(output, static_cast<std::uint32_t>(serializedRecordSize));
    writeLittleEndian(output, std::uint32_t{0});
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
        if (!lookup_.emplace(key(record.layer_id, record.expert_id), index).second) {
            throw StorageError("expert index contains a duplicate layer/expert key");
        }
    }
}

} // namespace hypermoe::storage
