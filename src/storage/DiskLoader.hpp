#pragma once

#include "storage/ExpertStore.hpp"

#include <future>
#include <memory>
#include <vector>

namespace hypermoe::storage {

enum class DiskReadMode {
    MemoryMap,
    RangeRead,
};

struct LoadedExpert {
    ExpertRecord record;
    std::vector<std::byte> bytes;
};

class DiskLoader {
public:
    explicit DiskLoader(std::shared_ptr<const ExpertStore> store,
                        DiskReadMode mode = DiskReadMode::MemoryMap);

    [[nodiscard]] LoadedExpert load(std::uint32_t layerId,
                                    std::uint32_t expertId) const;
    [[nodiscard]] std::future<LoadedExpert> loadAsync(std::uint32_t layerId,
                                                      std::uint32_t expertId) const;
    [[nodiscard]] const ExpertStore& store() const noexcept;

private:
    std::shared_ptr<const ExpertStore> store_;
    DiskReadMode mode_;
};

} // namespace hypermoe::storage
