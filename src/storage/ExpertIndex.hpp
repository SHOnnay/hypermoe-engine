#pragma once

#include <cstddef>
#include <array>
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

enum class ProjectionType : std::uint32_t { Gate = 0, Up = 1, Down = 2 };

struct ProjectionRecord {
    std::uint32_t layer_id{};
    std::uint32_t expert_id{};
    ProjectionType projection_type{ProjectionType::Gate};
    std::uint32_t dtype{};
    std::uint32_t rank{};
    std::uint32_t reserved{};
    std::uint64_t offset{};
    std::uint64_t size{};
    std::array<std::uint64_t, 4> shape{};
    std::uint32_t checksum{};
    std::uint32_t trailing_reserved{};

    friend bool operator==(const ProjectionRecord&, const ProjectionRecord&) = default;
};

class StorageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ExpertIndex {
public:
    static constexpr std::uint32_t formatVersion = 2;
    static constexpr std::uint32_t legacyFormatVersion = 1;
    static constexpr std::size_t serializedRecordSize = 32;
    static constexpr std::size_t serializedProjectionRecordSize = 80;

    ExpertIndex() = default;
    explicit ExpertIndex(std::vector<ExpertRecord> records);
    ExpertIndex(std::vector<ExpertRecord> records,
                std::vector<ProjectionRecord> projections);

    [[nodiscard]] static ExpertIndex load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path) const;

    [[nodiscard]] std::optional<ExpertRecord>
    find(std::uint32_t layerId, std::uint32_t expertId) const noexcept;
    [[nodiscard]] std::span<const ExpertRecord> records() const noexcept;
    [[nodiscard]] std::span<const ProjectionRecord> projections() const noexcept;
    [[nodiscard]] std::optional<ProjectionRecord> findProjection(
        std::uint32_t layerId,
        std::uint32_t expertId,
        ProjectionType type) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    [[nodiscard]] static std::uint64_t key(std::uint32_t layerId,
                                           std::uint32_t expertId) noexcept;
    void rebuildLookup();

    std::vector<ExpertRecord> records_;
    std::vector<ProjectionRecord> projections_;
    std::unordered_map<std::uint64_t, std::size_t> lookup_;
    std::unordered_map<std::uint64_t, std::array<std::size_t, 3>> projectionLookup_;
};

} // namespace hypermoe::storage
