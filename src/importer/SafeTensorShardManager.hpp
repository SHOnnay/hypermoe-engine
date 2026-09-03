#pragma once

#include "models/ModelManifest.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hypermoe::importer {

struct SafeTensorShard {
    std::filesystem::path relativePath;
    std::uint64_t fileSize{};
    std::size_t tensorCount{};
};

class SafeTensorShardManager {
public:
    explicit SafeTensorShardManager(const std::filesystem::path& artifact);

    [[nodiscard]] const std::filesystem::path& artifactRoot() const noexcept;
    [[nodiscard]] std::span<const SafeTensorShard> shards() const noexcept;
    [[nodiscard]] std::span<const models::ManifestTensor> tensors() const noexcept;
    [[nodiscard]] const models::ManifestTensor*
    find(std::string_view tensorName) const;
    [[nodiscard]] std::vector<std::byte>
    readTensor(std::string_view tensorName) const;
    [[nodiscard]] std::vector<std::byte>
    readRange(std::string_view tensorName,
              std::uint64_t relativeOffset,
              std::uint64_t size) const;

private:
    std::filesystem::path root_;
    std::vector<SafeTensorShard> shards_;
    std::vector<models::ManifestTensor> tensors_;
    std::unordered_map<std::string, std::size_t> lookup_;
};

} // namespace hypermoe::importer
