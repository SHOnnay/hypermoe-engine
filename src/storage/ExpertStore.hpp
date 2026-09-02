#pragma once

#include "storage/ExpertIndex.hpp"
#include "storage/MappedFile.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hypermoe::storage {

struct ExpertBlob {
    std::uint32_t layerId{};
    std::uint32_t expertId{};
    std::uint32_t quantizationType{};
    std::vector<std::byte> data;
};

class ExpertStore {
public:
    explicit ExpertStore(const std::filesystem::path& modelDirectory);

    static void create(const std::filesystem::path& modelDirectory,
                       std::span<const ExpertBlob> experts,
                       std::string_view metadataJson);

    [[nodiscard]] const ExpertIndex& index() const noexcept;
    [[nodiscard]] const std::string& metadataJson() const noexcept;
    [[nodiscard]] const std::filesystem::path& dataPath() const noexcept;

    [[nodiscard]] std::span<const std::byte>
    mappedExpert(std::uint32_t layerId,
                 std::uint32_t expertId,
                 bool validateChecksum = true) const;
    [[nodiscard]] std::vector<std::byte>
    readExpert(std::uint32_t layerId,
               std::uint32_t expertId,
               bool validateChecksum = true) const;

    [[nodiscard]] static std::uint32_t crc32(std::span<const std::byte> bytes) noexcept;

private:
    [[nodiscard]] ExpertRecord requireRecord(std::uint32_t layerId,
                                             std::uint32_t expertId) const;
    static void validate(std::span<const std::byte> bytes, const ExpertRecord& record);

    std::filesystem::path dataPath_;
    ExpertIndex index_;
    MappedFile mappedData_;
    std::string metadataJson_;
};

} // namespace hypermoe::storage
