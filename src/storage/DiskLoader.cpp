#include "storage/DiskLoader.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::storage {

DiskLoader::DiskLoader(std::shared_ptr<const ExpertStore> store, DiskReadMode mode)
    : store_(std::move(store)), mode_(mode) {
    if (!store_) {
        throw std::invalid_argument("DiskLoader requires an ExpertStore");
    }
}

LoadedExpert DiskLoader::load(std::uint32_t layerId, std::uint32_t expertId) const {
    const auto record = store_->index().find(layerId, expertId);
    if (!record) {
        throw StorageError("requested expert does not exist");
    }
    if (mode_ == DiskReadMode::RangeRead) {
        return {*record, store_->readExpert(layerId, expertId)};
    }
    const auto mapped = store_->mappedExpert(layerId, expertId);
    return {*record, std::vector<std::byte>(mapped.begin(), mapped.end())};
}

std::future<LoadedExpert>
DiskLoader::loadAsync(std::uint32_t layerId, std::uint32_t expertId) const {
    const auto store = store_;
    const auto mode = mode_;
    return std::async(std::launch::async,
                      [store, mode, layerId, expertId] {
                          return DiskLoader(store, mode).load(layerId, expertId);
                      });
}

const ExpertStore& DiskLoader::store() const noexcept {
    return *store_;
}

} // namespace hypermoe::storage
