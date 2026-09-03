#include "importer/SafeTensorShardManager.hpp"

#include "importer/SafeTensors.hpp"
#include "models/metadata/JsonValue.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <set>

namespace hypermoe::importer {
namespace {

using models::metadata::MetadataError;

bool safeRelative(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) return false;
    return std::none_of(path.begin(), path.end(), [](const auto& component) {
        return component == "..";
    });
}

} // namespace

SafeTensorShardManager::SafeTensorShardManager(
    const std::filesystem::path& artifact) {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(artifact, error);
    if (error) throw MetadataError("model artifact does not exist");
    const auto singleFile = std::filesystem::is_regular_file(canonical, error);
    if (error) throw MetadataError("cannot inspect model artifact");
    root_ = singleFile ? canonical.parent_path() : canonical;

    std::vector<std::filesystem::path> files;
    std::map<std::string, std::filesystem::path, std::less<>> declaredWeights;
    if (singleFile) {
        if (canonical.extension() != ".safetensors") {
            throw MetadataError("artifact file is not SafeTensors");
        }
        files.push_back(canonical);
    } else if (const auto indexPath = root_ / "model.safetensors.index.json";
               std::filesystem::exists(indexPath)) {
        const auto index = models::metadata::parseJsonFile(indexPath);
        std::set<std::filesystem::path> declaredShards;
        for (const auto& [name, shardValue] :
             index.require("weight_map").asObject()) {
            const std::filesystem::path declared = shardValue.asString();
            if (name.empty() || !safeRelative(declared) ||
                declared.extension() != ".safetensors") {
                throw MetadataError("SafeTensors shard index mapping is unsafe");
            }
            declaredWeights.emplace(name, declared.lexically_normal());
            declaredShards.insert(declared.lexically_normal());
        }
        if (declaredWeights.empty()) {
            throw MetadataError("SafeTensors shard index has no tensors");
        }
        for (const auto& shard : declaredShards) files.push_back(root_ / shard);
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(root_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                files.push_back(entry.path());
            }
        }
    }
    if (files.empty()) throw MetadataError("no SafeTensors shards found");
    std::sort(files.begin(), files.end());

    for (const auto& file : files) {
        auto shardTensors = SafeTensors::inspectFile(file, root_);
        const auto relative = std::filesystem::relative(file, root_, error);
        if (error) throw MetadataError("cannot resolve SafeTensors shard path");
        const auto fileSize = std::filesystem::file_size(file, error);
        if (error || !safeRelative(relative)) {
            throw MetadataError("cannot resolve SafeTensors shard");
        }
        shards_.push_back({relative.lexically_normal(),
                           static_cast<std::uint64_t>(fileSize),
                           shardTensors.size()});
        for (auto& tensor : shardTensors) {
            const auto index = tensors_.size();
            if (!lookup_.emplace(tensor.name, index).second) {
                throw MetadataError("duplicate tensor name across SafeTensors shards");
            }
            tensors_.push_back(std::move(tensor));
        }
    }

    if (!declaredWeights.empty()) {
        if (declaredWeights.size() != tensors_.size()) {
            throw MetadataError("SafeTensors shard index tensor count mismatch");
        }
        for (const auto& [name, declared] : declaredWeights) {
            const auto found = find(name);
            if (!found || found->sourceFile != declared) {
                throw MetadataError("SafeTensors shard index mapping is inconsistent");
            }
        }
    }
    std::sort(tensors_.begin(), tensors_.end(), [](const auto& left, const auto& right) {
        return left.name < right.name;
    });
    lookup_.clear();
    lookup_.reserve(tensors_.size());
    for (std::size_t index = 0; index < tensors_.size(); ++index) {
        lookup_.emplace(tensors_[index].name, index);
    }
}

const std::filesystem::path&
SafeTensorShardManager::artifactRoot() const noexcept { return root_; }

std::span<const SafeTensorShard>
SafeTensorShardManager::shards() const noexcept { return shards_; }

std::span<const models::ManifestTensor>
SafeTensorShardManager::tensors() const noexcept { return tensors_; }

const models::ManifestTensor* SafeTensorShardManager::find(
    std::string_view tensorName) const {
    const auto found = lookup_.find(std::string(tensorName));
    return found == lookup_.end() ? nullptr : &tensors_[found->second];
}

std::vector<std::byte> SafeTensorShardManager::readTensor(
    std::string_view tensorName) const {
    const auto* tensor = find(tensorName);
    if (!tensor) throw MetadataError("unknown SafeTensors tensor");
    return readRange(tensorName, 0, tensor->size);
}

std::vector<std::byte> SafeTensorShardManager::readRange(
    std::string_view tensorName,
    std::uint64_t relativeOffset,
    std::uint64_t size) const {
    const auto* tensor = find(tensorName);
    if (!tensor) throw MetadataError("unknown SafeTensors tensor");
    if (size == 0 || relativeOffset > tensor->size || size > tensor->size - relativeOffset ||
        tensor->offset > std::numeric_limits<std::uint64_t>::max() - relativeOffset ||
        size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw MetadataError("SafeTensors range request is invalid");
    }
    const auto absolute = tensor->offset + relativeOffset;
    if (absolute > static_cast<std::uint64_t>(
                       std::numeric_limits<std::streamoff>::max())) {
        throw MetadataError("SafeTensors range is not addressable");
    }
    std::ifstream input(root_ / tensor->sourceFile, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(absolute));
    std::vector<std::byte> result(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size()));
    if (!input) throw MetadataError("SafeTensors range read is truncated");
    return result;
}

} // namespace hypermoe::importer
