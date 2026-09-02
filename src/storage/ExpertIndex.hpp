#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hypermoe::storage {

struct ExpertRecord {
    std::uint32_t layer_id{};
    std::uint32_t expert_id{};
    std::uint64_t offset{};
    std::uint64_t size{};
    std::uint32_t quantization_type{};
    std::uint32_t checksum{};

    friend bool operator==(const ExpertRecord&, const ExpertRecord&) = default;
};

class StorageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ExpertIndex {
public:
    static constexpr std::uint32_t formatVersion = 1;
    static constexpr std::size_t serializedRecordSize = 32;

    ExpertIndex() = default;
    explicit ExpertIndex(std::vector<ExpertRecord> records);

    [[nodiscard]] static ExpertIndex load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path) const;

    [[nodiscard]] std::optional<ExpertRecord>
    find(std::uint32_t layerId, std::uint32_t expertId) const noexcept;
    [[nodiscard]] std::span<const ExpertRecord> records() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    [[nodiscard]] static std::uint64_t key(std::uint32_t layerId,
                                           std::uint32_t expertId) noexcept;
    void rebuildLookup();

    std::vector<ExpertRecord> records_;
    std::unordered_map<std::uint64_t, std::size_t> lookup_;
};

} // namespace hypermoe::storage
